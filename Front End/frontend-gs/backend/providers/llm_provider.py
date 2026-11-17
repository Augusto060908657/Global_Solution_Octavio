"""Acesso ao LLM. Único ponto do sistema que fala com o Ollama (local)."""
import logging
import os
from functools import lru_cache
from typing import Protocol

import httpx

logger = logging.getLogger("citizentech.llm")


class LLMError(Exception):
    """Erro base. `mensagem` é segura para exibir ao cidadão; `detalhe` só vai para o log."""

    codigo = "llm_erro"
    status_http = 502
    mensagem = "Não foi possível concluir a análise agora. Tente novamente."

    def __init__(self, detalhe: str = ""):
        super().__init__(detalhe or self.mensagem)
        self.detalhe = detalhe


class LLMUnavailableError(LLMError):
    codigo = "llm_indisponivel"
    status_http = 503
    mensagem = "O assistente de IA está indisponível no momento. Tente novamente em alguns minutos."


class LLMTimeoutError(LLMError):
    codigo = "llm_timeout"
    status_http = 504
    mensagem = "A análise demorou mais do que o esperado. Tente novamente ou resuma o relato."


class LLMInvalidResponseError(LLMError):
    codigo = "llm_resposta_invalida"
    status_http = 502
    mensagem = "Não conseguimos interpretar a resposta da IA. Tente reformular o relato."


class LLMProvider(Protocol):
    model: str

    def chat_json(self, messages: list[dict], schema: dict) -> str: ...
    def health(self) -> dict: ...


class OllamaProvider:
    """Cliente HTTP do Ollama com saída estruturada (parâmetro `format` = JSON Schema)."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.timeout = timeout or float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

    def chat_json(self, messages: list[dict], schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,                      # saída estruturada
            "keep_alive": "30m",                   # mantém o modelo na RAM (menos latência)
            "options": {"temperature": 0.1, "num_predict": 700},
        }
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"timeout após {self.timeout}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"Ollama inacessível em {self.base_url}: {exc}") from exc

        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Modelo '{self.model}' não instalado. Rode: ollama pull {self.model}")
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Ollama respondeu {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMInvalidResponseError(f"Envelope inesperado do Ollama: {exc}") from exc

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            nomes = [m.get("name", "") for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            instalado = any(n == self.model or (":" not in self.model and n.split(":")[0] == base)
                            for n in nomes)
            return {"ollama": True, "modelo": self.model, "modelo_instalado": instalado}
        except Exception as exc:  # noqa: BLE001 - health nunca deve lançar
            logger.info("Ollama health falhou: %s", exc)
            return {"ollama": False, "modelo": self.model, "modelo_instalado": False}


@lru_cache(maxsize=1)
def get_provider() -> OllamaProvider:
    return OllamaProvider()
