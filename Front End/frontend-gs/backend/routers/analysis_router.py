"""Endpoints HTTP. Sem lógica de IA aqui: apenas tradução HTTP <-> pipeline."""
import logging

from fastapi import APIRouter, Depends, HTTPException

from pipelines import llm_pipeline
from providers import feedback_store
from providers.llm_provider import LLMError, LLMProvider, get_provider
from schemas.citizen_schema import (CATEGORIA_LABELS, AnalysisRequest, AnalysisResponse,
                                    FeedbackRequest, FeedbackResponse, HealthResponse)

logger = logging.getLogger("citizentech.router")
router = APIRouter(prefix="/api/v1", tags=["CitizenTech"])


@router.get("/health", response_model=HealthResponse)
def health(provider: LLMProvider = Depends(get_provider)):
    estado = provider.health()
    ok = estado["ollama"] and estado["modelo_instalado"]
    return HealthResponse(status="ok" if ok else "degradado", **estado)


@router.get("/categorias")
def categorias() -> dict[str, str]:
    return CATEGORIA_LABELS


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest, provider: LLMProvider = Depends(get_provider)):
    try:
        return llm_pipeline.analyze(req, provider)
    except LLMError as exc:
        logger.error("Falha na análise (%s): %s", exc.codigo, exc.detalhe)   # técnico só no log
        raise HTTPException(status_code=exc.status_http,
                            detail={"codigo": exc.codigo, "mensagem": exc.mensagem})


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest):
    try:
        feedback_store.save_feedback(body.model_dump())
    except OSError as exc:
        logger.error("Não foi possível gravar feedback: %s", exc)
        raise HTTPException(status_code=500, detail={
            "codigo": "feedback_nao_salvo",
            "mensagem": "Não conseguimos salvar sua avaliação agora."})
    return FeedbackResponse(status="registrado", protocolo=body.protocolo)
