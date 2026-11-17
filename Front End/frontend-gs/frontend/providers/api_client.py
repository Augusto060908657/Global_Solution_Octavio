"""Cliente HTTP da API. Traduz TODA falha técnica em ApiError com mensagem amigável."""
import logging
import os

import requests

logger = logging.getLogger("citizentech.front.api")

_CAMPOS_ANALISE = {"protocolo", "criado_em", "categoria_label", "analise", "modelo", "latencia_ms"}
_CAMPOS_ANALISE_IA = {"categoria", "urgencia", "resumo", "orgao_responsavel", "orientacoes",
                      "prazo_estimado", "confianca"}


class ApiError(Exception):
    """Erro seguro para exibir ao cidadão (o detalhe técnico vai só para o log)."""

    def __init__(self, mensagem: str, codigo: str = "erro", sugestao: str = ""):
        super().__init__(mensagem)
        self.mensagem, self.codigo, self.sugestao = mensagem, codigo, sugestao


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout_analise: float | None = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.timeout_analise = timeout_analise or float(os.getenv("API_TIMEOUT_SECONDS", "120"))
        self.session = requests.Session()          # reaproveita conexão (ver @st.cache_resource)

    # ------------------------------------------------------------ núcleo
    def _request(self, method: str, path: str, timeout: float, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            logger.warning("Timeout em %s: %s", url, exc)
            raise ApiError("A análise está demorando mais que o normal.", "timeout",
                           "Tente novamente em instantes ou escreva um relato mais curto.") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.warning("API offline em %s: %s", url, exc)
            raise ApiError("Não conseguimos conectar ao serviço no momento.", "api_offline",
                           "Verifique sua conexão e tente novamente em alguns minutos.") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Erro de requisição: %s", exc)
            raise ApiError("Ocorreu um problema ao enviar sua solicitação.", "requisicao") from exc

        if resp.status_code >= 400:
            self._raise_http_error(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            logger.error("Resposta não-JSON de %s: %s", url, resp.text[:200])
            raise ApiError("Recebemos uma resposta inesperada do serviço.", "resposta_invalida",
                           "Tente novamente.") from exc
        return data

    @staticmethod
    def _raise_http_error(resp: requests.Response) -> None:
        detail = {}
        try:
            detail = resp.json().get("detail", {})
        except ValueError:
            pass
        logger.warning("HTTP %s: %s", resp.status_code, detail or resp.text[:200])
        if isinstance(detail, dict) and detail.get("mensagem"):
            raise ApiError(detail["mensagem"], detail.get("codigo", f"http_{resp.status_code}"),
                           "Se o problema continuar, tente novamente mais tarde.")
        raise ApiError("O serviço não conseguiu processar sua solicitação.", f"http_{resp.status_code}",
                       "Tente novamente em alguns minutos.")

    # ------------------------------------------------------------ endpoints
    def analyze(self, payload: dict) -> dict:
        data = self._request("POST", "/api/v1/analyze", self.timeout_analise, json=payload)
        if (not isinstance(data, dict) or not _CAMPOS_ANALISE <= data.keys()
                or not isinstance(data["analise"], dict) or not _CAMPOS_ANALISE_IA <= data["analise"].keys()):
            logger.error("Resposta da API fora do contrato: %s", str(data)[:300])
            raise ApiError("Recebemos uma resposta incompleta da análise.", "resposta_invalida",
                           "Tente novamente ou reformule o relato.")
        return data

    def send_feedback(self, payload: dict) -> dict:
        return self._request("POST", "/api/v1/feedback", 10, json=payload)

    def health(self) -> dict:
        return self._request("GET", "/api/v1/health", 4)

    def categorias(self) -> dict:
        return self._request("GET", "/api/v1/categorias", 5)
