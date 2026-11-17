"""Ponto de entrada da API CitizenTech (FastAPI)."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # execução local
load_dotenv()                                                   # Docker / cwd

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from routers.analysis_router import router  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("citizentech.api")


def _allowed_origins() -> list[str]:
    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
    if "*" in origins:
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise RuntimeError("ALLOWED_ORIGINS não pode conter '*' em produção.")
        logger.warning("CORS com '*' — use apenas em desenvolvimento.")
    return origins


app = FastAPI(title="CitizenTech API", version="1.0.0",
              description="Análise inteligente de solicitações do cidadão (Ollama + saída estruturada).")

app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins(), allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError):
    campos = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
    return JSONResponse(status_code=422, content={"detail": {
        "codigo": "entrada_invalida",
        "mensagem": "Confira os campos preenchidos e tente novamente.",
        "campos": campos}})


@app.exception_handler(Exception)
async def _unexpected_handler(_: Request, exc: Exception):
    logger.exception("Erro inesperado: %s", exc)
    return JSONResponse(status_code=500, content={"detail": {
        "codigo": "erro_interno", "mensagem": "Algo deu errado do nosso lado. Tente novamente."}})
