#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rpa_resilience.py - Validação e evidências de resiliência do robô RPA

Autor: Octavio Augusto Ramalho Brandão Pires - RM 555696

Duas formas de uso:
    pytest -v test_rpa_resilience.py      # testes automatizados
    python test_rpa_resilience.py         # demonstração + relatório de evidências
                                          # (gera evidencias/relatorio_validacao.json)

Todos os testes são offline e determinísticos: a API do clima é substituída
por funções falsas (mock) que simulam resposta normal ou serviço fora do ar.
"""

from __future__ import annotations

import json
import re
import smtplib
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import requests

import generate_test_data as gtd
import rpa_pipeline as rpa


# -----------------------------------------------------------------------------
# Dublês (mocks) da API de clima
# -----------------------------------------------------------------------------
class RespostaFalsa:
    """Simula requests.Response com tempo seco (clima neutro) ou chuva forte."""

    def __init__(self, chuva_dia: float = 0.0) -> None:
        self._chuva = chuva_dia

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {"current": {"precipitation": 0.0, "wind_gusts_10m": 10.0, "temperature_2m": 25.0},
                "daily": {"precipitation_sum": [self._chuva], "precipitation_probability_max": [10.0],
                          "wind_gusts_10m_max": [20.0]}}


def api_ok(*_a: Any, **_k: Any) -> RespostaFalsa:
    return RespostaFalsa()


def api_fora_do_ar(*_a: Any, **_k: Any) -> Any:
    raise requests.exceptions.ConnectTimeout("simulação: API de clima fora do ar")


def api_resposta_invalida(*_a: Any, **_k: Any) -> Any:
    class Ruim(RespostaFalsa):
        def json(self) -> Dict[str, Any]:
            return {"current": {}, "daily": None}      # JSON sem os campos esperados
    return Ruim()


def sem_espera(_segundos: float) -> None:
    """Substitui time.sleep para os testes rodarem instantaneamente."""


# -----------------------------------------------------------------------------
# Utilitários
# -----------------------------------------------------------------------------
def montar_ambiente(base: Path, **overrides: Any) -> tuple[rpa.Config, Dict[str, int]]:
    """Gera CSV de teste e uma Config isolada (db, log, cache e saídas próprios)."""
    csv_path = base / "chamados_cidadao_test.csv"
    esperado = gtd.gerar_csv(str(csv_path), seed=42)
    # Valores-padrão isolados do .env do desenvolvedor; `overrides` tem prioridade
    params: Dict[str, Any] = dict(
        csv_path=str(csv_path), output_dir=str(base / "out"), db_path=str(base / "out" / "t.db"),
        log_file=str(base / "rpa_execution.log"), cache_path=str(base / "cache" / "clima.json"),
        modo_email="simulado", api_backoff=0.0, api_retries=3, cb_threshold=3,
        smtp_host="", smtp_user="", smtp_password="", email_to=[])
    params.update(overrides)
    cfg = rpa.carregar_config(**params)
    return cfg, esperado


def medir_acuracia(cfg: rpa.Config) -> Dict[str, Any]:
    """Compara criticidade prevista x rótulo esperado nos registros válidos."""
    prior = pd.read_csv(Path(cfg.output_dir) / "chamados_priorizados.csv", dtype=str)
    gabarito = pd.read_csv(cfg.csv_path, dtype=str, keep_default_na=False, engine="python",
                           on_bad_lines="skip")[["id_chamado", "rotulo_esperado"]].drop_duplicates("id_chamado")
    junto = prior.merge(gabarito, on="id_chamado", how="left")
    acertos = int((junto["criticidade"] == junto["rotulo_esperado"]).sum())
    total = len(junto)
    erros = junto[junto["criticidade"] != junto["rotulo_esperado"]]
    ordem = ["BAIXA", "MEDIA", "ALTA", "CRITICA"]
    # erro "de segurança" = robô subestimou a gravidade (o mais perigoso no mundo real)
    subestimados = int(sum(ordem.index(r.criticidade) < ordem.index(r.rotulo_esperado)
                           for r in erros.itertuples()))
    # revisão humana capturou quantos dos erros?
    capturados = int((erros["revisao_humana"].astype(str) == "True").sum())
    return {"total": total, "acertos": acertos, "acuracia_pct": round(100 * acertos / total, 1) if total else 0.0,
            "erros": len(erros), "erros_por_subestimacao": subestimados,
            "erros_sinalizados_para_revisao_humana": capturados,
            "detalhe_erros": [{"id": r.id_chamado, "previsto": r.criticidade, "esperado": r.rotulo_esperado,
                               "categoria": r.categoria} for r in erros.itertuples()]}


# -----------------------------------------------------------------------------
# TESTES (pytest)
# -----------------------------------------------------------------------------
def test_taxa_de_processamento_e_isolamento_de_erros(tmp_path: Path) -> None:
    """100% dos chamados válidos processados; todos os corrompidos isolados; robô não cai."""
    cfg, esperado = montar_ambiente(tmp_path)
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert r["processados_novos"] == esperado["validos_esperados"]
    assert r["em_quarentena"] == esperado["invalidos_esperados"]
    assert r["linhas_lidas"] == esperado["total_linhas_dados"]
    assert r["logs_critical"] == 0
    assert r["fonte_clima"] == "api" and r["fallback_clima_acionado"] is False


def test_fallback_api_fora_do_ar_usa_cache_e_registra_log(tmp_path: Path) -> None:
    """API OK grava cache; depois API cai -> fallback para cache + registro no log."""
    cfg, _ = montar_ambiente(tmp_path)
    rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)             # popula cache
    cfg.db_path = str(tmp_path / "out" / "t2.db")                              # banco novo
    r = rpa.executar_pipeline(cfg, http_get=api_fora_do_ar, dormir=sem_espera)
    assert r["fonte_clima"] == "cache" and r["fallback_clima_acionado"] is True
    log = Path(cfg.log_file).read_text(encoding="utf-8")
    assert "FALLBACK_CACHE" in log and "FALLBACK acionado" in log
    assert r["processados_novos"] == 60                                        # robô seguiu até o fim


def test_fallback_sem_cache_segue_com_clima_neutro_e_revisao_humana(tmp_path: Path) -> None:
    """Pior caso: API fora e sem cache -> robô conclui e sinaliza revisão humana."""
    cfg, _ = montar_ambiente(tmp_path)
    r = rpa.executar_pipeline(cfg, http_get=api_fora_do_ar, dormir=sem_espera)
    assert r["fonte_clima"] == "indisponivel"
    assert r["processados_novos"] == 60 and r["revisao_humana"] > 0
    assert "FALLBACK_NEUTRO" in Path(cfg.log_file).read_text(encoding="utf-8")


def test_resposta_invalida_da_api_tambem_aciona_fallback(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    r = rpa.executar_pipeline(cfg, http_get=api_resposta_invalida, dormir=sem_espera)
    assert r["fonte_clima"] != "api" and r["processados_novos"] == 60


def test_retry_com_backoff_exponencial(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path, api_retries=3, api_backoff=1.0, cb_threshold=99)
    esperas: list[float] = []
    rpa.configurar_logging(cfg.log_file)
    rpa.coletar_clima(cfg, rpa.CircuitBreaker(99, 60), http_get=api_fora_do_ar, dormir=esperas.append)
    assert esperas == [1.0, 2.0]                         # 3 tentativas => 2 esperas dobrando


def test_circuit_breaker_abre_e_suprime_chamadas(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path, api_retries=5)
    chamadas = {"n": 0}

    def api_contada(*_a: Any, **_k: Any) -> Any:
        chamadas["n"] += 1
        raise requests.exceptions.ConnectionError("fora")

    rpa.configurar_logging(cfg.log_file)
    rpa.coletar_clima(cfg, rpa.CircuitBreaker(limite=2, cooldown=60), http_get=api_contada, dormir=sem_espera)
    assert chamadas["n"] == 2                            # 3ª tentativa bloqueada pelo circuito aberto
    assert "CIRCUITO_ABERTO" in Path(cfg.log_file).read_text(encoding="utf-8")


def test_circuit_breaker_fecha_apos_cooldown() -> None:
    t = {"agora": 0.0}
    cb = rpa.CircuitBreaker(limite=2, cooldown=10, relogio=lambda: t["agora"])
    cb.registrar_falha(); cb.registrar_falha()
    assert cb.aberto
    t["agora"] = 11.0
    assert cb.pode_chamar()


def test_arquivo_csv_ausente_nao_derruba_o_robo(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    cfg.csv_path = str(tmp_path / "nao_existe.csv")
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert r["csv_lido_com_sucesso"] is False and r["processados_novos"] == 0
    assert "ARQUIVO_AUSENTE" in Path(cfg.log_file).read_text(encoding="utf-8")
    assert (Path(cfg.output_dir) / "resumo_execucao.json").exists()   # relatório gerado mesmo assim


def test_csv_com_schema_invalido(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    ruim = tmp_path / "ruim.csv"
    ruim.write_text("coluna_x,coluna_y\n1,2\n", encoding="utf-8")
    cfg.csv_path = str(ruim)
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert r["processados_novos"] == 0 and "SCHEMA_INVALIDO" in Path(cfg.log_file).read_text(encoding="utf-8")


def test_idempotencia_reexecucao_nao_duplica_nem_realerta(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    r1 = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    r2 = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert r1["processados_novos"] == 60 and r1["alertas_criticos"] > 0
    assert r2["processados_novos"] == 0 and r2["ja_processados_anteriormente"] == 60
    assert r2["alertas_criticos"] == 0


def test_falha_smtp_preserva_mensagem_em_outbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _ = montar_ambiente(tmp_path, modo_email="smtp", smtp_host="smtp.invalido.test",
                             smtp_user="u", smtp_password="p", smtp_retries=2)

    def smtp_quebrado(*_a: Any, **_k: Any) -> Any:
        raise ConnectionRefusedError("simulação: SMTP recusou conexão")

    monkeypatch.setattr(rpa.smtplib, "SMTP", smtp_quebrado)
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert set(r["status_emails"].values()) == {"falha_outbox"}
    assert len(list((Path(cfg.output_dir) / "outbox").glob("*.eml"))) == 2
    assert "FALLBACK_OUTBOX" in Path(cfg.log_file).read_text(encoding="utf-8")


def test_smtp_sem_credenciais_nao_quebra(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path, modo_email="smtp", smtp_host="", smtp_user="", smtp_password="")
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert set(r["status_emails"].values()) == {"falha_outbox"}


def test_lgpd_nenhum_dado_pessoal_nas_saidas(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    r = rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    assert r["dados_pessoais_mascarados"] == 6
    saidas = "".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in Path(cfg.output_dir).rglob("*") if p.suffix in {".csv", ".eml", ".json", ".md"})
    assert not re.search(r"@exemplo\.(com|org)", saidas)
    assert not re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", saidas)
    assert "91234-5678" not in saidas and "98765-4321" not in saidas


def test_acuracia_minima_da_priorizacao(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    rpa.executar_pipeline(cfg, http_get=api_ok, dormir=sem_espera)
    m = medir_acuracia(cfg)
    assert m["total"] == 60 and m["acuracia_pct"] >= 70.0


def test_alerta_climatico_dispara_com_chuva_forte(tmp_path: Path) -> None:
    cfg, _ = montar_ambiente(tmp_path)
    r = rpa.executar_pipeline(cfg, http_get=lambda *a, **k: RespostaFalsa(chuva_dia=80.0), dormir=sem_espera)
    assert r["alerta_climatico"] is True and r["alertas_disparados"] == r["alertas_criticos"] + 1


# -----------------------------------------------------------------------------
# DEMONSTRAÇÃO (python test_rpa_resilience.py)
# -----------------------------------------------------------------------------
def main() -> int:
    """Executa os 3 cenários e imprime/grava o relatório de evidências."""
    base = Path(tempfile.mkdtemp(prefix="rpa_validacao_"))
    evidencias: Dict[str, Any] = {}

    print("\n=== CENÁRIO 1: operação normal (API respondendo) ===")
    (base / "c1").mkdir()
    cfg1, esperado = montar_ambiente(base / "c1")
    r1 = rpa.executar_pipeline(cfg1, http_get=api_ok, dormir=sem_espera)
    acur = medir_acuracia(cfg1)
    taxa = round(100 * r1["processados_novos"] / esperado["validos_esperados"], 1)

    print("\n=== CENÁRIO 2: API DE CLIMA FORA DO AR (cache disponível) -> FALLBACK ===")
    cfg2 = rpa.carregar_config(**{**vars(cfg1), "db_path": str(base / "c1" / "out" / "cenario2.db"),
                                  "log_file": str(base / "c1" / "cenario2.log")})
    r2 = rpa.executar_pipeline(cfg2, http_get=api_fora_do_ar, dormir=sem_espera)

    print("\n=== CENÁRIO 3: API FORA DO AR e SEM CACHE ===")
    cfg3 = rpa.carregar_config(**{**vars(cfg1), "db_path": str(base / "c1" / "out" / "cenario3.db"),
                                  "log_file": str(base / "c1" / "cenario3.log"),
                                  "cache_path": str(base / "c1" / "sem_cache.json")})
    r3 = rpa.executar_pipeline(cfg3, http_get=api_fora_do_ar, dormir=sem_espera)

    evidencias = {
        "amostra": esperado,
        "taxa_processamento_validos_pct": taxa,
        "acuracia_priorizacao": acur,
        "cenario_1_normal": r1, "cenario_2_fallback_cache": r2, "cenario_3_fallback_neutro": r3,
    }
    pasta = Path("evidencias")
    pasta.mkdir(exist_ok=True)
    (pasta / "relatorio_validacao.json").write_text(
        json.dumps(evidencias, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70 + "\nRELATÓRIO DE VALIDAÇÃO\n" + "=" * 70)
    print(f"Amostra: {esperado['total_linhas_dados']} linhas "
          f"({esperado['validos_esperados']} válidas esperadas, {esperado['invalidos_esperados']} inválidas esperadas)")
    print(f"Taxa de processamento dos válidos: {taxa}% ({r1['processados_novos']}/{esperado['validos_esperados']})")
    print(f"Erros isolados (quarentena): {r1['em_quarentena']} de {esperado['invalidos_esperados']} esperados")
    print(f"Alertas disparados: {r1['alertas_disparados']} (críticos={r1['alertas_criticos']})")
    print(f"Acurácia da criticidade: {acur['acertos']}/{acur['total']} = {acur['acuracia_pct']}% | "
          f"subestimações: {acur['erros_por_subestimacao']} | erros já sinalizados p/ revisão humana: "
          f"{acur['erros_sinalizados_para_revisao_humana']}/{acur['erros']}")
    print(f"Fallback cenário 2: fonte={r2['fonte_clima']} | cenário 3: fonte={r3['fonte_clima']} "
          f"(processados {r2['processados_novos']} e {r3['processados_novos']})")
    print("Evidências gravadas em evidencias/relatorio_validacao.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
