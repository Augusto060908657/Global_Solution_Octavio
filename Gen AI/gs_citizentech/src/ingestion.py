"""
BLOCO 1 — Ingestão e estratégia de chunking.

Base de conhecimento: manual de serviços do "Cidade Para Todos" (corpus de demonstração em
./data/*.md). Cada arquivo traz um front matter com os metadados oficiais (título, categoria,
URL de origem, data de atualização e data de consulta). Para usar fontes reais (gov.br, Cartas de
Serviços), basta salvar a página como .md/.txt em ./data com o mesmo front matter.

ESTRATÉGIA DE CHUNKING: HÍBRIDA (estrutura + tamanho)
  1) Divisão por ESTRUTURA: MarkdownHeaderTextSplitter separa o documento por cabeçalhos (#, ##).
     Cada seção de uma carta de serviço trata de UM assunto (prazo, documentos, como solicitar),
     então cortar nas fronteiras de seção evita misturar assuntos num mesmo vetor.
  2) Divisão por TAMANHO: seções maiores que `chunk_size` são subdivididas com
     RecursiveCharacterTextSplitter (parágrafo > linha > frase > palavra).
  3) Contexto herdado: o prefixo "Título > Seção" é adicionado ao texto de cada chunk. Assim,
     um trecho como "O prazo é de 10 dias úteis" carrega a informação de QUE serviço é, o que
     melhora tanto a recuperação (embedding) quanto a citação (LLM).

POR QUE chunk_size=500 e chunk_overlap=50?
  * Limite do encoder: paraphrase-multilingual-MiniLM-L12-v2 trunca a entrada em 128 tokens
    (~450-550 caracteres em português). Chunks de 500 caracteres cabem inteiros no embedding;
    chunks maiores teriam a cauda ignorada e perderiam evidências (ex.: a exceção ao prazo).
  * Granularidade da resposta: ~500 caracteres (~80 palavras) comportam um tópico completo
    (ex.: "prazo + exceção prioritária") sem diluir o sinal semântico com assuntos vizinhos.
  * Overlap de 50 (10%): quando uma seção precisa ser subdividida, a sobreposição repete a
    última frase do trecho anterior no seguinte, evitando que uma regra seja cortada ao meio
    (ex.: condição num chunk e prazo no outro). Mais que ~15% geraria redundância no top-k.
  * Janela de contexto da LLM: com k=3, o contexto fica em ~1.500 caracteres + prefixos
    (< 600 tokens), muito abaixo da janela do gerador (8k-128k tokens): sobra espaço para o
    system prompt, o histórico e a resposta, reduzindo custo e o efeito "lost in the middle".
  A avaliação (evaluate_rag.py) compara 500/50 com 300/30, 800/80 e com chunking fixo puro.
"""
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config


@dataclass
class Document:
    doc_id: str
    body: str
    metadata: Dict[str, str]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_front_matter(raw: str, default_title: str) -> tuple:
    """Lê o bloco `--- chave: valor ---` no início do arquivo (sem depender de PyYAML)."""
    meta = {
        "titulo": default_title,
        "categoria": "geral",
        "url_origem": "não informada",
        "data_atualizacao": "não informada",
        "data_consulta": "não informada",
    }
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        return meta, raw
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, match.group(2)


def load_documents(data_dir: Optional[Path] = None) -> List[Document]:
    """Carrega todos os .md/.txt da pasta de dados preservando os metadados oficiais."""
    data_dir = Path(data_dir or config.DATA_DIR)
    files = sorted(list(data_dir.glob("*.md")) + list(data_dir.glob("*.txt")))
    if not files:
        raise FileNotFoundError(f"Nenhum documento .md/.txt encontrado em {data_dir}")
    documents = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw, default_title=path.stem)
        meta["source"] = path.name
        documents.append(Document(doc_id=path.stem, body=body.strip(), metadata=meta))
    return documents


def _recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )


def chunk_documents(
    documents: List[Document],
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """Gera os chunks com metadados. strategy='hybrid' (cabeçalhos+tamanho) ou 'fixed' (só tamanho)."""
    strategy = strategy or config.CHUNK_STRATEGY
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    recursive = _recursive_splitter(chunk_size, chunk_overlap)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo_doc"), ("##", "secao")], strip_headers=True
    )

    chunks: List[Chunk] = []
    for doc in documents:
        pieces = []  # (secao, texto)
        if strategy == "hybrid":
            for section in header_splitter.split_text(doc.body):
                secao = section.metadata.get("secao", "Introdução")
                titulo = section.metadata.get("titulo_doc", doc.metadata["titulo"])
                for piece in recursive.split_text(section.page_content):
                    pieces.append((secao, f"{titulo} > {secao}\n{piece.strip()}"))
        elif strategy == "fixed":
            for piece in recursive.split_text(doc.body):
                pieces.append(("(sem seção)", piece.strip()))
        else:
            raise ValueError(f"Estratégia de chunking desconhecida: {strategy}")

        for idx, (secao, text) in enumerate(pieces):
            if not text.strip():
                continue
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "doc_id": doc.doc_id,
                    "secao": secao,
                    "chunk_index": str(idx),
                    "chunking": f"{strategy}-{chunk_size}/{chunk_overlap}",
                }
            )
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}-{idx:03d}", text=text, metadata=metadata))
    return chunks


def ingest(strategy=None, chunk_size=None, chunk_overlap=None) -> List[Chunk]:
    return chunk_documents(load_documents(), strategy, chunk_size, chunk_overlap)


if __name__ == "__main__":
    docs = load_documents()
    all_chunks = chunk_documents(docs)
    sizes = [len(c.text) for c in all_chunks]
    print(f"Documentos: {len(docs)} | Chunks: {len(all_chunks)} | "
          f"tamanho médio: {sum(sizes) / len(sizes):.0f} | máx: {max(sizes)} caracteres")
    print(f"Configuração: {config.CHUNK_STRATEGY} {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}\n")
    for c in all_chunks[:3]:
        print(f"[{c.chunk_id}] {c.metadata['titulo']} | {c.metadata['categoria']} | "
              f"{c.metadata['url_origem']} | {c.metadata['data_atualizacao']}")
        print(c.text, "\n")
