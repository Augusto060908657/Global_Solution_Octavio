"""Testes unitários do pipeline e da API — rodam sem Ollama e sem interface (pytest)."""
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from pipelines import llm_pipeline
from providers.llm_provider import (LLMInvalidResponseError, LLMTimeoutError,
                                    LLMUnavailableError, get_provider)
from schemas.citizen_schema import AnalysisRequest, LLMAnalysis

VALID = {
    "categoria": "iluminacao_publica", "urgencia": "baixa",
    "resumo": "Poste apagado na rua.", "justificativa_urgencia": "Sem risco imediato.",
    "orgao_responsavel": "Prefeitura – iluminação pública",
    "orientacoes": ["Registre na central 156."], "prazo_estimado": "5 a 15 dias úteis",
    "risco_imediato": False, "confianca": 0.9,
}


class FakeProvider:
    model = "fake-model"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def chat_json(self, messages, schema):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)

    def health(self):
        return {"ollama": True, "modelo": self.model, "modelo_instalado": True}


def req(texto="A lâmpada do poste da minha rua está queimada há dias"):
    return AnalysisRequest(relato=texto)


# ---------------------------------------------------------------- pipeline
def test_mask_pii():
    texto, mascarou = llm_pipeline.mask_pii("Meu CPF 123.456.789-09, email a@b.com, tel (11) 91234-5678")
    assert mascarou and "123.456" not in texto and "a@b.com" not in texto and "91234" not in texto


def test_pipeline_ok():
    resp = llm_pipeline.analyze(req(), FakeProvider([VALID]))
    assert resp.protocolo.startswith("CT-") and resp.analise.categoria == "iluminacao_publica"
    assert resp.categoria_label == "Iluminação pública" and not resp.alerta_seguranca


def test_pipeline_retry_after_invalid_json():
    prov = FakeProvider(["isto não é json", VALID])
    assert llm_pipeline.analyze(req(), prov).analise.urgencia == "baixa" and prov.calls == 2


def test_pipeline_invalid_twice_raises():
    with pytest.raises(LLMInvalidResponseError):
        llm_pipeline.analyze(req(), FakeProvider(["lixo", {"categoria": "inexistente"}]))


def test_safety_rule_overrides_low_urgency():
    r = AnalysisRequest(relato="Sinto cheiro de gás forte no corredor do prédio e ninguém resolve")
    resp = llm_pipeline.analyze(r, FakeProvider([VALID]))
    assert resp.alerta_seguranca and resp.analise.urgencia == "critica" and resp.analise.risco_imediato
    assert any("193" in t for t in resp.telefones_emergencia)


def test_llm_analysis_normalizes_confidence_and_accents():
    a = LLMAnalysis.model_validate({**VALID, "confianca": 85, "urgencia": "Média"})
    assert a.confianca == 0.85 and a.urgencia == "media"


def test_pii_not_sent_to_llm():
    msgs = llm_pipeline.build_messages(llm_pipeline.mask_pii("CPF 123.456.789-09 poste apagado")[0], None, None, None)
    assert "123.456" not in msgs[-1]["content"]


# ---------------------------------------------------------------- API
@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def _use(provider):
    app.dependency_overrides[get_provider] = lambda: provider


def test_api_rejects_short_report(client):
    r = client.post("/api/v1/analyze", json={"relato": "curto"})
    assert r.status_code == 422 and r.json()["detail"]["codigo"] == "entrada_invalida"


def test_api_rejects_bad_cep(client):
    r = client.post("/api/v1/analyze", json={"relato": "Buraco enorme na rua principal do bairro", "cep": "123"})
    assert r.status_code == 422


def test_api_analyze_ok(client):
    _use(FakeProvider([VALID]))
    r = client.post("/api/v1/analyze", json={"relato": "Poste apagado na minha rua há duas semanas"})
    assert r.status_code == 200 and r.json()["analise"]["categoria"] == "iluminacao_publica"


@pytest.mark.parametrize("erro,status,codigo", [
    (LLMUnavailableError("x"), 503, "llm_indisponivel"),
    (LLMTimeoutError("x"), 504, "llm_timeout"),
])
def test_api_llm_errors_are_friendly(client, erro, status, codigo):
    _use(FakeProvider([erro]))
    r = client.post("/api/v1/analyze", json={"relato": "Poste apagado na minha rua há duas semanas"})
    assert r.status_code == status and r.json()["detail"]["codigo"] == codigo
    assert "Traceback" not in r.text and "x" != r.json()["detail"]["mensagem"]


def test_api_feedback(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    r = client.post("/api/v1/feedback", json={"protocolo": "CT-20260101-ABC123", "decisao": "confirmado",
                                              "avaliacao": "util"})
    assert r.status_code == 200 and (tmp_path / "feedback.jsonl").exists()


def test_cors_is_restrictive(client):
    ok = client.options("/api/v1/analyze", headers={"Origin": "http://localhost:8501",
                                                    "Access-Control-Request-Method": "POST"})
    bad = client.options("/api/v1/analyze", headers={"Origin": "http://site-malicioso.com",
                                                     "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in bad.headers


def test_health_degraded_when_ollama_down(client):
    class Down(FakeProvider):
        def health(self):
            return {"ollama": False, "modelo": "m", "modelo_instalado": False}
    _use(Down([]))
    assert client.get("/api/v1/health").json()["status"] == "degradado"
