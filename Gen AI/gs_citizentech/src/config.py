"""Configuração central do módulo de IA Generativa do CitizenTech "Cidade Para Todos".

Todos os parâmetros podem ser sobrescritos por variáveis de ambiente (ou arquivo .env).
Nenhuma credencial é escrita no código: a chave da OpenAI é lida de OPENAI_API_KEY.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv é opcional
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = BASE_DIR / "resultados"

# ----------------------------- Identificação do grupo -------------------------
# (preencha aqui se o grupo tiver mais integrantes; aparece no relatório gerado e no vídeo)
AUTHORS = [("Octavio Augusto Ramalho Brandão Pires", "555696")]
COURSE = "FIAP · Global Solution 2026.2 · Generative AI & Advanced Nets · Turma 2TIAPF"

# ----------------------------- Chunking ---------------------------------------
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "hybrid")  # hybrid | fixed
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ----------------------------- Embeddings -------------------------------------
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ----------------------------- Retrieval --------------------------------------
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | hybrid
TOP_K = int(os.getenv("TOP_K", "3"))
# None => usa o padrão do backend de embeddings (ver retrieval.Embedder.default_threshold)
_thr = os.getenv("SIM_THRESHOLD")
SIM_THRESHOLD = float(_thr) if _thr else None
RRF_K = 60  # constante da Reciprocal Rank Fusion
CANDIDATE_POOL = 20

# ----------------------------- Generator --------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")  # gratuito, local, roda em CPU/GPU modesta
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "180"))  # 1ª chamada do Ollama carrega o modelo na memória
LLM_TEMPERATURE = 0.0  # determinismo: reduz variação e alucinação
LLM_MAX_TOKENS = 450

# ----------------------------- Segurança / UX ---------------------------------
MAX_QUESTION_CHARS = 500
HISTORY_TURNS = 3
EXTRACTIVE_MIN_COVERAGE = 0.45  # cobertura lexical mínima no modo extractive
GENERIC_TERM_MIN_DOCS = 3  # termo único presente em >= N documentos é "genérico" (pede esclarecimento)
SERVICE_EXAMPLES = ("reparo de buraco, iluminação pública, coleta de móveis e entulho, poda de árvore, "
                    "calçadas e rampas, gratuidade no transporte, credencial de vaga, IPTU ou ouvidoria")

OFFICIAL_CHANNEL = (
    "a Central 156 (telefone ou WhatsApp), o portal cidadeparatodos.example.gov.br "
    "ou um Centro de Atendimento ao Cidadão"
)
