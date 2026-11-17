"""
BLOCO 2 — Armazenamento vetorial (FAISS) e Retriever híbrido com Top-K estrito e limiar.

* Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilíngue, 384 dim,
  roda em CPU). Se o pacote/modelo não estiver disponível (ex.: sem internet na 1ª execução), usa
  um fallback leve (HashingVectorizer de n-gramas de caracteres) para a demo continuar funcionando;
  o backend usado é sempre exibido no log.
* Índice: faiss.IndexFlatIP sobre vetores L2-normalizados => produto interno = similaridade de cosseno
  (busca exata; para poucos milhares de chunks é a escolha mais simples e sem perda de recall).
* Retriever HÍBRIDO: união dos candidatos densos (FAISS) e lexicais (BM25) combinados por
  Reciprocal Rank Fusion (RRF). O léxico recupera termos exatos (siglas, "NBR 9050", "156"); o denso
  recupera paráfrases ("conserto de buraco" ~ "reparo de pavimento").
* TOP-K ESTRITO (k=3) + LIMIAR: o que decide se um trecho é relevante é a similaridade de cosseno
  DENSA (score absoluto, comparável entre consultas). Abaixo do limiar o trecho é descartado, e se
  nada passa o retriever devolve lista vazia => o gerador é obrigado a se abster.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # fallback de demonstração
    SentenceTransformer = None

import config
from ingestion import Chunk
from text_utils import strip_pii_tags, tokenize


class Embedder:
    """Encapsula o modelo de embeddings, com fallback explícito e visível."""

    def __init__(self, backend: Optional[str] = None, model_name: Optional[str] = None):
        backend = backend or config.EMBEDDING_BACKEND
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.backend = "hashing"
        self.model = None
        if backend in ("sentence-transformers", "st", "auto"):
            if SentenceTransformer is None:
                print("[AVISO] sentence-transformers não instalado: usando fallback 'hashing'.")
            else:
                try:
                    self.model = SentenceTransformer(self.model_name)
                    self.backend = "sentence-transformers"
                except Exception as exc:  # sem internet / modelo indisponível
                    print(f"[AVISO] Falha ao carregar '{self.model_name}' ({exc.__class__.__name__}). "
                          "Usando fallback 'hashing'.")
        if self.backend == "hashing":
            # Fallback lexical: unigramas de palavras (sem stopwords, com stemming) em espaço hash.
            self._vectorizer = HashingVectorizer(
                analyzer=tokenize, n_features=4096, alternate_sign=False, norm="l2",
            )
        if self.model is not None:
            getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
            self.dim = getter()
        else:
            self.dim = 4096

    @property
    def default_threshold(self) -> float:
        return 0.30 if self.backend == "sentence-transformers" else 0.10

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
            return np.ascontiguousarray(vecs, dtype="float32")
        return np.ascontiguousarray(self._vectorizer.transform(texts).toarray(), dtype="float32")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # similaridade de cosseno densa (0-1) — base do limiar e da confiança
    bm25: float
    rank: int


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, mode: Optional[str] = None,
                 k: Optional[int] = None, threshold: Optional[float] = None):
        self.chunks = chunks
        self.embedder = embedder
        self.mode = mode or config.RETRIEVAL_MODE
        self.k = k or config.TOP_K
        if threshold is not None:
            self.threshold = threshold
        elif config.SIM_THRESHOLD is not None:
            self.threshold = config.SIM_THRESHOLD
        else:
            self.threshold = embedder.default_threshold

        vectors = embedder.encode([c.text for c in chunks])
        self.vectors = vectors  # guardados em memória: evita reconstruct() do FAISS
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens)
        df = Counter(t for toks in self._tokens for t in set(toks))
        n = len(chunks)
        self.idf: Dict[str, float] = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.idf_max = math.log(n + 1) + 1  # peso de um termo que não existe na base

    # --------------------------------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None, mode: Optional[str] = None,
                 threshold: Optional[float] = None) -> List[RetrievedChunk]:
        k = k or self.k
        mode = mode or self.mode
        threshold = self.threshold if threshold is None else threshold
        n = len(self.chunks)
        pool = min(n, config.CANDIDATE_POOL)

        query = strip_pii_tags(query)  # etiquetas de PII não fazem parte da intenção da pergunta
        qvec = self.embedder.encode([query])
        scores, ids = self.index.search(qvec, pool)
        dense = {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}
        dense_rank = {i: r for r, i in enumerate(dense)}

        bm_scores = np.zeros(n)
        bm_rank: Dict[int, int] = {}
        if mode == "hybrid":
            q_tokens = tokenize(query)
            if q_tokens:
                bm_scores = self.bm25.get_scores(q_tokens)
                order = [int(i) for i in np.argsort(-bm_scores)[:pool] if bm_scores[i] > 0]
                bm_rank = {i: r for r, i in enumerate(order)}
                for i in order:  # candidatos só-lexicais também recebem score denso
                    if i not in dense:
                        dense[i] = float(np.dot(qvec[0], self.vectors[i]))

        if mode == "hybrid":
            fused = {}
            for i in dense:
                value = 0.0
                if i in dense_rank:
                    value += 1.0 / (config.RRF_K + dense_rank[i] + 1)
                if i in bm_rank:
                    value += 1.0 / (config.RRF_K + bm_rank[i] + 1)
                fused[i] = value
            ranking = sorted(fused, key=lambda i: fused[i], reverse=True)
        else:
            ranking = sorted(dense, key=lambda i: dense[i], reverse=True)

        results: List[RetrievedChunk] = []
        for i in ranking:
            if dense[i] < threshold:  # Similarity Threshold Cutoff
                continue
            results.append(RetrievedChunk(self.chunks[i], dense[i], float(bm_scores[i]), len(results) + 1))
            if len(results) == k:  # Top-K estrito
                break
        return results

    # --------------------------------------------------------------------------------------
    def lexical_coverage(self, query: str, retrieved: List[RetrievedChunk]) -> float:
        """Fração (ponderada por IDF) dos termos da pergunta presentes nos trechos recuperados.
        Termos inexistentes na base recebem peso máximo: pergunta sobre algo que a base não
        conhece (ex.: 'CNH') tem cobertura baixa mesmo que o tema pareça próximo."""
        q_terms = set(tokenize(query))
        if not q_terms:
            return 0.0
        ctx_terms = set()
        for r in retrieved:
            ctx_terms.update(tokenize(r.chunk.text))
        weights = {t: self.idf.get(t, self.idf_max) for t in q_terms}
        covered = sum(w for t, w in weights.items() if t in ctx_terms)
        return covered / sum(weights.values())

    def doc_frequency(self, token: str) -> int:
        """Em quantos DOCUMENTOS distintos o termo aparece (termo genérico = aparece em vários)."""
        return len({c.metadata["doc_id"] for c, toks in zip(self.chunks, self._tokens) if token in toks})

    def idf_weight(self, token: str) -> float:
        return self.idf.get(token, self.idf_max)


def build_retriever(chunks: List[Chunk], embedder: Optional[Embedder] = None, **kwargs) -> HybridRetriever:
    return HybridRetriever(chunks, embedder or Embedder(), **kwargs)


if __name__ == "__main__":
    import sys
    from ingestion import ingest

    retriever = build_retriever(ingest())
    print(f"Backend: {retriever.embedder.backend} | modo: {retriever.mode} | k={retriever.k} | "
          f"limiar={retriever.threshold}")
    query = " ".join(sys.argv[1:]) or "Qual o prazo para consertar um buraco na rua?"
    print(f"Pergunta: {query}")
    for r in retriever.retrieve(query):
        print(f"  #{r.rank} score={r.score:.3f} bm25={r.bm25:.2f} [{r.chunk.chunk_id}] "
              f"{r.chunk.text[:90]!r}")
