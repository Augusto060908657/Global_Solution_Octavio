"""
BLOCO 4 — API FastAPI para integração com o front-end do CitizenTech.

Executar:  uvicorn api_service:app --reload --port 8000
Docs interativas (Swagger): http://localhost:8000/docs   |   Chat de demonstração: http://localhost:8000/
"""
import hashlib
import logging
import os
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Deque, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from rag_chain import RAGChain, build_chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("citizentech.api")

MAX_SESSIONS = 1000
_sessions: "OrderedDict[str, Deque[str]]" = OrderedDict()  # histórico curto por sessão (só em memória)
_state = {}


class UserQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS,
                          description="Pergunta do cidadão", examples=["Qual o prazo para reparar um buraco?"])
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                            description="Identificador anônimo da sessão (não use dados pessoais)",
                            examples=["sessao-demo-001"])


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    abstained: bool
    processing_time_sec: float
    # campos adicionais, opcionais para o front-end
    blocked: bool = False
    clarification: bool = False
    service_error: bool = False
    security_status: str = "ok"


@asynccontextmanager
async def lifespan(app: FastAPI):
    chain = build_chain()  # carrega embeddings, FAISS e gerador uma única vez
    _state["chain"] = chain
    log.info("Pronto | embeddings=%s | gerador=%s | chunks=%d", chain.retriever.embedder.backend,
             chain.llm.provider, len(chain.retriever.chunks))
    yield


app = FastAPI(title="CitizenTech — Assistente Cidade Para Todos (RAG)", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8000").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


def _history(session_id: str) -> Deque[str]:
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=config.HISTORY_TURNS)
        if len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    _sessions.move_to_end(session_id)
    return _sessions[session_id]


@app.get("/health")
def health():
    chain: RAGChain = _state["chain"]
    return {"status": "ok", "embeddings": chain.retriever.embedder.backend, "generator": chain.llm.provider,
            "chunks": len(chain.retriever.chunks)}


@app.post("/api/v1/chat", response_model=RAGResponse)
def chat(query: UserQuery) -> RAGResponse:
    chain: RAGChain = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Assistente ainda inicializando.")
    history = _history(query.session_id)
    result = chain.answer(query.question, history=list(history))
    if not result.blocked:
        history.append(result.sanitized_question)  # guarda a versão já com PII mascarada
    # Privacidade: o log não registra o texto da pergunta, apenas um hash curto e metadados.
    log.info("chat sessao=%s q_hash=%s abstained=%s blocked=%s motivo=%s t=%.2fs",
             hashlib.sha256(query.session_id.encode()).hexdigest()[:8],
             hashlib.sha256(result.sanitized_question.encode()).hexdigest()[:8],
             result.abstained, result.blocked, result.reason, result.processing_time_sec)
    return RAGResponse(answer=result.answer, sources=result.sources,
                       confidence_score=result.confidence_score, abstained=result.abstained,
                       processing_time_sec=round(result.processing_time_sec, 3),
                       blocked=result.blocked, clarification=result.clarification, service_error=result.service_error, security_status=result.security_status)


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(config.STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_service:app", host="127.0.0.1", port=8000, reload=False)
