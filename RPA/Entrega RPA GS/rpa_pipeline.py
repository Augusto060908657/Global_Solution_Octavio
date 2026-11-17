#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rpa_pipeline.py - Robô RPA + IA da plataforma CitizenTech "Cidade Para Todos"
Global Solution FIAP 2026.2 - AI for Robotic Process Automation

Autor: Octavio Augusto Ramalho Brandão Pires - RM 555696

FLUXO PONTA A PONTA
    ENTRADA   : (1) API pública Open-Meteo (clima)   -> Tema 1
                (2) CSV de chamados de cidadãos      -> Tema 2
    PROCESSO  : validação de schema, sanitização (LGPD), quarentena de
                registros inválidos, priorização por regras inteligentes
                (léxico NLP + categoria + clima), persistência em SQLite/CSV
                                                      -> Tema 2
    SAÍDA     : alerta de emergência acessível + resumo de zeladoria por
                e-mail (SMTP real ou simulado) + relatório de execução
                                                      -> Tema 3

RESILIÊNCIA
    * timeout + retry com espera exponencial + circuit breaker na API
    * fallback para cache local quando a API está fora do ar
    * quarentena de registros corrompidos (o robô nunca para)
    * falha de SMTP -> mensagem preservada em outbox/ (caminho seguro)
    * idempotência: reexecutar não duplica registros nem reenvia alertas
    * revisão humana sinalizada para o que o robô não resolve com segurança

Uso:
    python rpa_pipeline.py                      # usa valores do .env
    python rpa_pipeline.py --csv meu.csv --modo-email simulado
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import smtplib
import sqlite3
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# =============================================================================
# 1. CONFIGURAÇÃO (credenciais e parâmetros SEMPRE via variáveis de ambiente)
# =============================================================================

logger = logging.getLogger("rpa")

COLUNAS_OBRIGATORIAS = [
    "id_chamado", "data_registro", "categoria", "descricao", "latitude", "longitude",
]

# Categorias de ocorrência aceitas pelo robô (demais vão para quarentena)
CATEGORIAS = {
    "buraco", "semaforo_quebrado", "rampa_obstruida", "alagamento",
    "iluminacao_publica", "queda_arvore", "calcada_danificada",
}

# Caixa geográfica aproximada do município de São Paulo (validação de coordenadas)
BBOX_SP = {"lat_min": -24.10, "lat_max": -23.30, "lon_min": -46.90, "lon_max": -46.30}


def _env_int(nome: str, padrao: int) -> int:
    """Lê inteiro do ambiente; se inválido, devolve o padrão (nunca levanta)."""
    try:
        return int(os.getenv(nome, str(padrao)))
    except ValueError:
        return padrao


def _env_float(nome: str, padrao: float) -> float:
    """Lê float do ambiente; se inválido, devolve o padrão (nunca levanta)."""
    try:
        return float(os.getenv(nome, str(padrao)))
    except ValueError:
        return padrao


@dataclass
class Config:
    """Todos os parâmetros do robô. Segredos vêm do .env (nunca do código)."""
    # --- Entrada ---
    csv_path: str = "chamados_cidadao_test.csv"
    api_url: str = "https://api.open-meteo.com/v1/forecast"
    latitude: float = -23.5505
    longitude: float = -46.6333
    api_timeout: float = 5.0
    api_retries: int = 3
    api_backoff: float = 1.0          # espera inicial (s); dobra a cada tentativa
    cb_threshold: int = 3             # falhas consecutivas para abrir o circuito
    cb_cooldown: float = 60.0         # segundos com o circuito aberto
    cache_path: str = "cache/clima_cache.json"
    # --- Saída ---
    output_dir: str = "outputs"
    db_path: str = "outputs/cidade_para_todos.db"
    log_file: str = "rpa_execution.log"
    # --- E-mail ---
    modo_email: str = "simulado"      # "simulado" (grava .eml) ou "smtp"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_timeout: float = 10.0
    smtp_retries: int = 2
    email_from: str = "robo@cidadeparatodos.exemplo.invalid"
    email_to: List[str] = field(default_factory=list)
    # --- Regra de priorização ---
    limiar_critica: int = 75
    limiar_alta: int = 55
    limiar_media: int = 40


def carregar_config(**overrides: Any) -> Config:
    """Carrega .env e monta a Config. `overrides` permite sobrescrever (testes/CLI)."""
    load_dotenv()
    destinatarios = [e.strip() for e in os.getenv("EMAIL_TO", "").split(",") if e.strip()]
    cfg = Config(
        csv_path=os.getenv("CSV_PATH", "chamados_cidadao_test.csv"),
        api_url=os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast"),
        latitude=_env_float("CITY_LAT", -23.5505),
        longitude=_env_float("CITY_LON", -46.6333),
        api_timeout=_env_float("API_TIMEOUT", 5.0),
        api_retries=_env_int("API_RETRIES", 3),
        api_backoff=_env_float("API_BACKOFF", 1.0),
        cb_threshold=_env_int("CB_THRESHOLD", 3),
        cb_cooldown=_env_float("CB_COOLDOWN", 60.0),
        cache_path=os.getenv("CACHE_PATH", "cache/clima_cache.json"),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
        db_path=os.getenv("DB_PATH", "outputs/cidade_para_todos.db"),
        log_file=os.getenv("LOG_FILE", "rpa_execution.log"),
        modo_email=os.getenv("MODO_EMAIL", "simulado").strip().lower(),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=_env_int("SMTP_PORT", 587),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_timeout=_env_float("SMTP_TIMEOUT", 10.0),
        smtp_retries=_env_int("SMTP_RETRIES", 2),
        email_from=os.getenv("EMAIL_FROM", "robo@cidadeparatodos.exemplo.invalid"),
        email_to=destinatarios,
    )
    for chave, valor in overrides.items():
        if not hasattr(cfg, chave):
            raise AttributeError(f"Parâmetro de configuração desconhecido: {chave}")
        setattr(cfg, chave, valor)
    return cfg


# =============================================================================
# 2. LOGS ESTRUTURADOS (arquivo + console, 4 níveis, timestamp/etapa/status)
# =============================================================================

class ContadorDeNiveis(logging.Handler):
    """Handler que apenas conta quantos logs de cada nível ocorreram (vai ao relatório)."""

    def __init__(self) -> None:
        super().__init__()
        self.contagem: Dict[str, int] = {"INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelname in self.contagem:
            self.contagem[record.levelname] += 1


def configurar_logging(log_file: str) -> ContadorDeNiveis:
    """Configura logger 'rpa' com arquivo (rpa_execution.log) e console."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):          # evita handlers duplicados em reexecuções
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
    formato = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formato)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formato)
    contador = ContadorDeNiveis()
    for h in (fh, ch, contador):
        logger.addHandler(h)
    return contador


def registrar(nivel: int, etapa: str, status: str, mensagem: str) -> None:
    """Log padronizado: [ETAPA] status=... | mensagem  (timestamp e nível vêm do formatter)."""
    logger.log(nivel, "[%s] status=%s | %s", etapa, status, mensagem)


# =============================================================================
# 3. TEMA 1 - CONSUMO DE API PÚBLICA (Open-Meteo) + RESILIÊNCIA
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker simples: após `limite` falhas consecutivas o circuito ABRE e
    novas chamadas são bloqueadas por `cooldown` segundos (evita martelar um
    serviço fora do ar). Um sucesso fecha o circuito e zera as falhas.
    """

    def __init__(self, limite: int = 3, cooldown: float = 60.0,
                 relogio: Callable[[], float] = time.monotonic) -> None:
        self.limite = limite
        self.cooldown = cooldown
        self._relogio = relogio
        self.falhas = 0
        self._aberto_ate = 0.0

    @property
    def aberto(self) -> bool:
        return self._relogio() < self._aberto_ate

    def pode_chamar(self) -> bool:
        return not self.aberto

    def registrar_sucesso(self) -> None:
        self.falhas = 0
        self._aberto_ate = 0.0

    def registrar_falha(self) -> None:
        self.falhas += 1
        if self.falhas >= self.limite:
            self._aberto_ate = self._relogio() + self.cooldown


def _interpretar_open_meteo(payload: Dict[str, Any]) -> Dict[str, float]:
    """Extrai e valida os campos usados. Levanta KeyError/TypeError/ValueError se inválido."""
    atual, dia = payload["current"], payload["daily"]
    clima = {
        "precipitacao_atual_mm": float(atual["precipitation"]),
        "rajada_atual_kmh": float(atual["wind_gusts_10m"]),
        "temperatura_c": float(atual["temperature_2m"]),
        "precipitacao_dia_mm": float(dia["precipitation_sum"][0]),
        "prob_chuva_pct": float(dia["precipitation_probability_max"][0]),
        "rajada_dia_kmh": float(dia["wind_gusts_10m_max"][0]),
    }
    for nome, valor in clima.items():
        if not math.isfinite(valor):
            raise ValueError(f"valor não finito em {nome}")
        if nome != "temperatura_c" and valor < 0:
            raise ValueError(f"valor negativo em {nome}")
    return clima


def _salvar_cache(caminho: str, dados: Dict[str, float]) -> None:
    """Grava cache de forma atômica (arquivo temporário + replace)."""
    try:
        destino = Path(caminho)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(".tmp")
        tmp.write_text(json.dumps({"coletado_em": datetime.now().isoformat(timespec="seconds"),
                                   "dados": dados}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(destino)
    except OSError as exc:
        registrar(logging.WARNING, "CLIMA", "CACHE_NAO_SALVO", f"não foi possível gravar cache: {exc}")


def _ler_cache(caminho: str) -> Optional[Dict[str, Any]]:
    """Lê cache; devolve None se ausente/corrompido (nunca levanta)."""
    try:
        bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
        dados = {k: float(v) for k, v in bruto["dados"].items()}
        coletado = datetime.fromisoformat(bruto["coletado_em"])
        return {"dados": dados, "coletado_em": coletado}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        registrar(logging.WARNING, "CLIMA", "CACHE_INDISPONIVEL", f"cache ausente ou inválido: {exc}")
        return None


def coletar_clima(config: Config, breaker: CircuitBreaker,
                  http_get: Callable[..., Any] = requests.get,
                  dormir: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """
    Coleta o clima (Open-Meteo). Ordem de tentativa:
      1) API com timeout + retry (backoff exponencial) + circuit breaker
      2) FALLBACK: último valor em cache local
      3) FALLBACK final: clima neutro marcado como 'indisponivel' (revisão humana)
    Sempre devolve um dicionário; nunca levanta exceção.
    """
    params = {
        "latitude": config.latitude, "longitude": config.longitude,
        "current": "precipitation,wind_gusts_10m,temperature_2m",
        "daily": "precipitation_sum,precipitation_probability_max,wind_gusts_10m_max",
        "timezone": "America/Sao_Paulo", "forecast_days": 1,
    }
    ultimo_erro = "desconhecido"
    for tentativa in range(1, config.api_retries + 1):
        if not breaker.pode_chamar():
            registrar(logging.WARNING, "CLIMA", "CIRCUITO_ABERTO",
                      "circuit breaker aberto: chamada à API suprimida")
            ultimo_erro = "circuit breaker aberto"
            break
        try:
            resp = http_get(config.api_url, params=params, timeout=config.api_timeout)
            resp.raise_for_status()
            clima = _interpretar_open_meteo(resp.json())
            breaker.registrar_sucesso()
            _salvar_cache(config.cache_path, clima)
            registrar(logging.INFO, "CLIMA", "OK", f"API respondeu na tentativa {tentativa}: "
                      f"chuva_dia={clima['precipitacao_dia_mm']}mm rajada_dia={clima['rajada_dia_kmh']}km/h")
            return {"fonte": "api", "idade_horas": 0.0, **clima}
        except (requests.exceptions.RequestException, ValueError, KeyError,
                TypeError, IndexError) as exc:
            breaker.registrar_falha()
            ultimo_erro = f"{type(exc).__name__}: {str(exc)[:90]}"
            registrar(logging.WARNING, "CLIMA", "FALHA_API",
                      f"tentativa {tentativa}/{config.api_retries} falhou ({ultimo_erro})")
            if tentativa < config.api_retries and breaker.pode_chamar():
                espera = config.api_backoff * (2 ** (tentativa - 1))
                registrar(logging.INFO, "CLIMA", "RETRY", f"aguardando {espera:.1f}s antes de nova tentativa")
                dormir(espera)

    # ---- FALLBACK 1: cache local ----
    cache = _ler_cache(config.cache_path)
    if cache is not None:
        idade_h = (datetime.now() - cache["coletado_em"]).total_seconds() / 3600
        nivel = logging.WARNING
        registrar(nivel, "CLIMA", "FALLBACK_CACHE",
                  f"FALLBACK acionado: API indisponível ({ultimo_erro}); usando cache com {idade_h:.1f}h de idade")
        if idade_h > 24:
            registrar(logging.WARNING, "CLIMA", "CACHE_ANTIGO", "cache com mais de 24h: decisões climáticas menos confiáveis")
        return {"fonte": "cache", "idade_horas": round(idade_h, 1), **cache["dados"]}

    # ---- FALLBACK 2: clima neutro + revisão humana ----
    registrar(logging.ERROR, "CLIMA", "FALLBACK_NEUTRO",
              f"FALLBACK acionado: API ({ultimo_erro}) e cache indisponíveis; seguindo sem fator climático")
    return {"fonte": "indisponivel", "idade_horas": None,
            "precipitacao_atual_mm": 0.0, "rajada_atual_kmh": 0.0, "temperatura_c": 0.0,
            "precipitacao_dia_mm": 0.0, "prob_chuva_pct": 0.0, "rajada_dia_kmh": 0.0}


def ha_alerta_climatico(clima: Dict[str, Any]) -> bool:
    """Critério objetivo de alerta climático de emergência (só com dado confiável)."""
    if clima.get("fonte") == "indisponivel":
        return False
    return (clima["precipitacao_dia_mm"] >= 50 or clima["precipitacao_atual_mm"] >= 10
            or clima["rajada_dia_kmh"] >= 70)


# =============================================================================
# 4. TEMA 2 - INGESTÃO, SANITIZAÇÃO E VALIDAÇÃO COM PANDAS
# =============================================================================

_RE_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_RE_TEL = re.compile(r"(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}")


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sem acentos e sem espaços nas pontas (para léxico e categorias)."""
    nfkd = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def anonimizar_texto(texto: str) -> Tuple[str, int]:
    """LGPD: mascara CPF, e-mail e telefone no texto livre. Devolve (texto, nº de máscaras)."""
    total = 0
    for padrao, marca in ((_RE_CPF, "[CPF]"), (_RE_EMAIL, "[EMAIL]"), (_RE_TEL, "[TELEFONE]")):
        texto, n = padrao.subn(marca, texto)
        total += n
    return texto, total


def carregar_chamados(config: Config) -> Tuple[pd.DataFrame, List[Dict[str, Any]], bool]:
    """
    Lê o CSV de chamados. Devolve (dataframe, linhas_malformadas, leitura_ok).
    Trata: arquivo ausente, encoding errado, linhas com nº de colunas errado,
    arquivo vazio e colunas obrigatórias faltando. Nunca levanta exceção.
    """
    vazio = pd.DataFrame(columns=COLUNAS_OBRIGATORIAS)
    caminho = Path(config.csv_path)
    if not caminho.exists():
        registrar(logging.ERROR, "ENTRADA_CSV", "ARQUIVO_AUSENTE",
                  f"arquivo '{caminho}' não encontrado; execução segue sem chamados")
        return vazio, [], False

    malformadas: List[Dict[str, Any]] = []

    def _linha_ruim(campos: List[str]) -> None:       # chamado pelo pandas p/ linhas inválidas
        malformadas.append({"linha_bruta": ";".join(map(str, campos))})
        return None                                      # None => descarta a linha

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        malformadas.clear()
        try:
            df = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding=encoding,
                             engine="python", on_bad_lines=_linha_ruim)
            if encoding != "utf-8-sig":
                registrar(logging.WARNING, "ENTRADA_CSV", "ENCODING", "UTF-8 falhou; lido como latin-1")
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
            registrar(logging.ERROR, "ENTRADA_CSV", "LEITURA_FALHOU", f"{type(exc).__name__}: {exc}")
            return vazio, [], False
    if df is None:
        registrar(logging.ERROR, "ENTRADA_CSV", "ENCODING_INVALIDO", "não foi possível decodificar o arquivo")
        return vazio, [], False

    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        registrar(logging.ERROR, "ENTRADA_CSV", "SCHEMA_INVALIDO",
                  f"colunas obrigatórias ausentes: {faltantes}; nenhum chamado processado")
        return vazio, malformadas, False

    registrar(logging.INFO, "ENTRADA_CSV", "OK",
              f"{len(df)} linhas lidas, {len(malformadas)} linha(s) malformada(s) descartada(s)")
    return df, malformadas, True


def _parse_data(valor: str, agora: datetime) -> Optional[datetime]:
    """ISO-8601 ('2026-11-06 14:30:00' ou '2026-11-06'). Rejeita inválida e futura."""
    try:
        data = datetime.fromisoformat(str(valor).strip())
    except ValueError:
        return None
    if data > agora:
        return None
    return data


def _parse_coord(valor: str, minimo: float, maximo: float) -> Optional[float]:
    """Converte coordenada; None se vazia, não numérica, não finita ou fora do intervalo."""
    try:
        x = float(str(valor).strip().replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(x) or not (minimo <= x <= maximo):
        return None
    return x


def validar_chamados(df: pd.DataFrame, agora: Optional[datetime] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Valida registro a registro (schema + regras de negócio). Um registro ruim NUNCA
    derruba o robô: vai para a quarentena com o(s) motivo(s).
    Devolve (validos, quarentena, nº_de_mascaras_LGPD).
    """
    agora = agora or datetime.now()
    validos: List[Dict[str, Any]] = []
    quarentena: List[Dict[str, Any]] = []
    ids_vistos: set = set()
    total_mascaras = 0

    for _, linha in df.iterrows():
        motivos: List[str] = []
        try:
            id_ = str(linha["id_chamado"]).strip()
            if not id_:
                motivos.append("id_ausente")
            elif id_ in ids_vistos:
                motivos.append("id_duplicado")

            data = _parse_data(linha["data_registro"], agora)
            if data is None:
                motivos.append("data_invalida_ou_futura")

            lat = _parse_coord(linha["latitude"], BBOX_SP["lat_min"], BBOX_SP["lat_max"])
            lon = _parse_coord(linha["longitude"], BBOX_SP["lon_min"], BBOX_SP["lon_max"])
            if lat is None or lon is None:
                motivos.append("coordenadas_ausentes_ou_invalidas")

            categoria = normalizar_texto(linha["categoria"]).replace(" ", "_")
            if categoria not in CATEGORIAS:
                motivos.append("categoria_desconhecida")

            descricao = str(linha["descricao"]).strip()
            if len(descricao) < 8 or descricao.lower() in {"nan", "null", "none", "n/a"}:
                motivos.append("descricao_nula_ou_curta")

            if motivos:
                quarentena.append({**{c: linha.get(c, "") for c in COLUNAS_OBRIGATORIAS},
                                   "motivo": ";".join(motivos)})
                continue

            descricao_limpa, n_mask = anonimizar_texto(descricao)
            total_mascaras += n_mask
            ids_vistos.add(id_)
            bairro = str(linha.get("bairro", "")).strip() or "Não informado"
            validos.append({"id_chamado": id_, "data_registro": data.isoformat(sep=" "),
                            "categoria": categoria, "descricao": descricao_limpa,
                            "latitude": lat, "longitude": lon, "bairro": bairro})
        except Exception as exc:  # noqa: BLE001 - última barreira: isola o registro
            quarentena.append({**{c: str(linha.get(c, "")) for c in COLUNAS_OBRIGATORIAS},
                               "motivo": f"erro_inesperado:{type(exc).__name__}"})
            registrar(logging.ERROR, "VALIDACAO", "REGISTRO_ISOLADO",
                      f"erro inesperado ao validar registro: {exc}")

    df_validos = pd.DataFrame(validos)
    df_quarentena = pd.DataFrame(quarentena)
    registrar(logging.INFO, "VALIDACAO", "OK",
              f"{len(df_validos)} válidos, {len(df_quarentena)} em quarentena, "
              f"{total_mascaras} dado(s) pessoal(is) mascarado(s) (LGPD)")
    if len(df_quarentena):
        resumo = df_quarentena["motivo"].str.split(";").explode().value_counts().to_dict()
        registrar(logging.WARNING, "VALIDACAO", "QUARENTENA", f"motivos: {resumo}")
    return df_validos, df_quarentena, total_mascaras


# =============================================================================
# 5. IA POR REGRAS INTELIGENTES - PRIORIZAÇÃO DE CRITICIDADE
# =============================================================================

PESO_BASE = {
    "alagamento": 45, "queda_arvore": 45, "semaforo_quebrado": 40, "rampa_obstruida": 35,
    "buraco": 30, "iluminacao_publica": 25, "calcada_danificada": 25,
}
CATEGORIAS_SENSIVEIS_AO_CLIMA = {"alagamento", "buraco", "queda_arvore", "iluminacao_publica"}

# Léxico NLP: (radical normalizado, peso). Combina risco à vida, vulnerabilidade
# (acessibilidade é o foco do "Cidade Para Todos"), urgência e atenuantes.
LEXICO: List[Tuple[str, int]] = [
    # risco à vida / integridade
    ("acident", 25), ("ferid", 25), ("choque", 20), ("fio exposto", 20), ("poste caid", 15),
    ("cratera", 15), ("desabament", 25), ("ambulanc", 15),
    # vulnerabilidade e acessibilidade
    ("cadeirante", 20), ("cadeira de rodas", 20), ("cego", 15), ("deficien", 15),
    ("idos", 15), ("crianc", 10), ("gestante", 10), ("escola", 10), ("hospital", 15),
    # urgência / bloqueio
    ("urgent", 10), ("perigo", 10), ("bloquead", 10), ("bloqueio", 10), ("obstruid", 10),
    ("impedid", 10), ("ilhad", 15), ("intransit", 15),
    # atenuantes
    ("pequen", -10), ("leve", -10), ("pintura", -10), ("poca", -10), ("estetic", -15),
]
_LEXICO_RE = [(re.compile(r"\b" + re.escape(r)), p, r) for r, p in LEXICO]
TETO_LEXICO = 45


def pontuar_chamado(chamado: Dict[str, Any], clima: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """
    Calcula score 0-100 e criticidade. Retorna também os motivos (explicabilidade)
    e a flag de revisão humana quando o robô não deve decidir sozinho.
    """
    categoria = chamado["categoria"]
    texto = normalizar_texto(chamado["descricao"])
    motivos = [f"categoria {categoria} (+{PESO_BASE[categoria]})"]

    # 1) léxico (NLP por regras), com teto para evitar inflação
    pts_lex, termos = 0, []
    for regex, peso, radical in _LEXICO_RE:
        if regex.search(texto):
            pts_lex += peso
            termos.append(f"'{radical}' ({peso:+d})")
    pts_lex = max(-20, min(TETO_LEXICO, pts_lex))
    if termos:
        motivos.append("termos: " + ", ".join(termos))

    # 2) fator climático (só com dado confiável: API ou cache)
    pts_clima = 0
    if clima.get("fonte") != "indisponivel" and categoria in CATEGORIAS_SENSIVEIS_AO_CLIMA:
        if clima["precipitacao_dia_mm"] >= 50 or clima["precipitacao_atual_mm"] >= 5:
            pts_clima = 15
        elif clima["prob_chuva_pct"] >= 70 or clima["precipitacao_dia_mm"] >= 20:
            pts_clima = 8
        if categoria == "queda_arvore" and clima["rajada_dia_kmh"] >= 60:
            pts_clima += 10
        if pts_clima:
            motivos.append(f"clima adverso (+{pts_clima})")

    score = max(0, min(100, PESO_BASE[categoria] + pts_lex + pts_clima))
    if score >= config.limiar_critica:
        criticidade = "CRITICA"
    elif score >= config.limiar_alta:
        criticidade = "ALTA"
    elif score >= config.limiar_media:
        criticidade = "MEDIA"
    else:
        criticidade = "BAIXA"

    # 3) revisão humana: o que o robô não resolve com segurança
    razoes_revisao = []
    if any(abs(score - lim) <= 2 for lim in (config.limiar_critica, config.limiar_alta, config.limiar_media)):
        razoes_revisao.append("pontuação na fronteira entre faixas")
    if clima.get("fonte") == "indisponivel" and categoria in CATEGORIAS_SENSIVEIS_AO_CLIMA:
        razoes_revisao.append("sem dado climático confiável")
    return {"score": score, "criticidade": criticidade, "motivos": " | ".join(motivos),
            "revisao_humana": bool(razoes_revisao), "motivo_revisao": "; ".join(razoes_revisao)}


def priorizar(df_validos: pd.DataFrame, clima: Dict[str, Any], config: Config) -> pd.DataFrame:
    """Aplica a regra a todos os chamados válidos (falha isolada por registro)."""
    if df_validos.empty:
        return df_validos.assign(score=[], criticidade=[], motivos=[], revisao_humana=[], motivo_revisao=[])
    resultados = []
    for _, linha in df_validos.iterrows():
        try:
            resultados.append(pontuar_chamado(linha.to_dict(), clima, config))
        except Exception as exc:  # noqa: BLE001
            registrar(logging.ERROR, "PRIORIZACAO", "REGISTRO_ISOLADO",
                      f"{linha['id_chamado']}: {exc}; marcado para revisão humana")
            resultados.append({"score": 0, "criticidade": "BAIXA", "motivos": "erro na priorização",
                               "revisao_humana": True, "motivo_revisao": "erro na priorização"})
    out = pd.concat([df_validos.reset_index(drop=True), pd.DataFrame(resultados)], axis=1)
    ordem = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3}
    out = out.sort_values(by=["criticidade", "score"], key=lambda s: s.map(ordem) if s.name == "criticidade" else -s)
    registrar(logging.INFO, "PRIORIZACAO", "OK",
              f"distribuição: {out['criticidade'].value_counts().to_dict()}")
    return out.reset_index(drop=True)


# =============================================================================
# 6. PERSISTÊNCIA (SQLite) E IDEMPOTÊNCIA
# =============================================================================

def persistir(df: pd.DataFrame, config: Config, execucao_id: str) -> pd.DataFrame:
    """
    Grava em SQLite com INSERT OR IGNORE (chave primária = id_chamado):
    reexecutar o robô NÃO duplica registros e NÃO reenvia alertas (idempotência).
    Devolve apenas os chamados NOVOS. Em caso de erro de banco, segue sem
    idempotência (todos tratados como novos) e registra ERROR.
    """
    if df.empty:
        return df
    try:
        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
        novos_ids = []
        with sqlite3.connect(config.db_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS chamados_processados (
                id_chamado TEXT PRIMARY KEY, data_registro TEXT, categoria TEXT, descricao TEXT,
                latitude REAL, longitude REAL, bairro TEXT, score INTEGER, criticidade TEXT,
                motivos TEXT, revisao_humana INTEGER, processado_em TEXT, execucao_id TEXT)""")
            for _, r in df.iterrows():
                cur = con.execute(
                    "INSERT OR IGNORE INTO chamados_processados VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["id_chamado"], r["data_registro"], r["categoria"], r["descricao"], r["latitude"],
                     r["longitude"], r["bairro"], int(r["score"]), r["criticidade"], r["motivos"],
                     int(bool(r["revisao_humana"])), datetime.now().isoformat(timespec="seconds"), execucao_id))
                if cur.rowcount == 1:
                    novos_ids.append(r["id_chamado"])
        novos = df[df["id_chamado"].isin(novos_ids)].reset_index(drop=True)
        ja_existiam = len(df) - len(novos)
        registrar(logging.INFO, "PERSISTENCIA", "OK",
                  f"{len(novos)} novo(s) gravado(s) no SQLite; {ja_existiam} já existiam (ignorados)")
        return novos
    except sqlite3.Error as exc:
        registrar(logging.ERROR, "PERSISTENCIA", "FALHA_BANCO",
                  f"{type(exc).__name__}: {exc}; seguindo sem idempotência (apenas CSV)")
        return df


# =============================================================================
# 7. TEMA 3 - NOTIFICAÇÕES E RELATÓRIOS (SMTP / simulador)
# =============================================================================

CSS_ACESSIVEL = ("body{font-family:Arial,Helvetica,sans-serif;font-size:18px;line-height:1.5;"
                 "color:#111;background:#fff;} h1{font-size:26px;} h2{font-size:21px;} "
                 "table{border-collapse:collapse;width:100%;} th,td{border:2px solid #333;padding:8px;"
                 "text-align:left;vertical-align:top;} th{background:#e6e6e6;} "
                 ".aviso{border:3px solid #000;padding:12px;font-weight:bold;}")


def _tabela_html(df: pd.DataFrame) -> str:
    """Tabela HTML semântica (th scope) — a criticidade aparece como TEXTO, não só cor."""
    cab = ("<tr><th scope='col'>Criticidade</th><th scope='col'>ID</th><th scope='col'>Tipo</th>"
           "<th scope='col'>Bairro</th><th scope='col'>Local (lat, lon)</th><th scope='col'>Descrição</th></tr>")
    linhas = "".join(
        f"<tr><td>{html.escape(r.criticidade)} ({int(r.score)})</td><td>{html.escape(r.id_chamado)}</td>"
        f"<td>{html.escape(r.categoria.replace('_', ' '))}</td><td>{html.escape(r.bairro)}</td>"
        f"<td>{r.latitude:.5f}, {r.longitude:.5f}</td><td>{html.escape(r.descricao)}</td></tr>"
        for r in df.itertuples())
    return f"<table>{cab}{linhas}</table>"


def montar_alerta_emergencia(criticos: pd.DataFrame, clima: Dict[str, Any],
                             alerta_clima: bool, config: Config) -> Optional[EmailMessage]:
    """E-mail de ALERTA para equipes de campo/gestores (texto simples + HTML acessível)."""
    if criticos.empty and not alerta_clima:
        return None
    msg = EmailMessage()
    msg["Subject"] = f"[ALERTA DE EMERGÊNCIA] Cidade Para Todos - {len(criticos)} chamado(s) crítico(s)"
    msg["From"] = config.email_from
    msg["To"] = ", ".join(config.email_to) or "gestores@cidadeparatodos.exemplo.invalid"

    texto = ["ALERTA DE EMERGÊNCIA - CIDADE PARA TODOS", ""]
    bloco_clima_html = ""
    if alerta_clima:
        linha = (f"ALERTA CLIMÁTICO: chuva prevista {clima['precipitacao_dia_mm']} mm, "
                 f"rajadas de até {clima['rajada_dia_kmh']} km/h (fonte: {clima['fonte']}).")
        texto += [linha, ""]
        bloco_clima_html = f"<p class='aviso'>{html.escape(linha)}</p>"
    if not criticos.empty:
        texto.append("Chamados de criticidade CRÍTICA que exigem ação imediata:")
        for r in criticos.itertuples():
            texto.append(f"- {r.id_chamado} | {r.categoria.replace('_', ' ')} | {r.bairro} | "
                         f"({r.latitude:.5f}, {r.longitude:.5f}) | {r.descricao}")
    texto += ["", "Mensagem automática. Dados anonimizados; nenhum dado pessoal é enviado."]
    msg.set_content("\n".join(texto))
    corpo = (f"<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'><title>Alerta de emergência</title>"
             f"<style>{CSS_ACESSIVEL}</style></head><body><main><h1>Alerta de emergência - Cidade Para Todos</h1>"
             f"{bloco_clima_html}"
             f"{'<h2>Chamados críticos</h2>' + _tabela_html(criticos) if not criticos.empty else ''}"
             f"<p>Mensagem automática. Dados anonimizados; nenhum dado pessoal é enviado.</p></main></body></html>")
    msg.add_alternative(corpo, subtype="html")
    return msg


def montar_resumo_zeladoria(df: pd.DataFrame, resumo: Dict[str, Any], clima: Dict[str, Any],
                            config: Config) -> EmailMessage:
    """E-mail de resumo operacional para a equipe de zeladoria."""
    msg = EmailMessage()
    msg["Subject"] = f"[Zeladoria] Resumo da execução {resumo['execucao_id']} - {resumo['processados_novos']} chamado(s)"
    msg["From"] = config.email_from
    msg["To"] = ", ".join(config.email_to) or "zeladoria@cidadeparatodos.exemplo.invalid"
    por_crit = df["criticidade"].value_counts().to_dict() if not df.empty else {}
    por_cat = df["categoria"].value_counts().to_dict() if not df.empty else {}
    revisao = df[df["revisao_humana"]] if not df.empty else df
    top = df.head(10)

    linhas = [f"RESUMO DE ZELADORIA - execução {resumo['execucao_id']}", "",
              f"Chamados lidos: {resumo['linhas_lidas']} | processados (novos): {resumo['processados_novos']} | "
              f"em quarentena: {resumo['em_quarentena']}",
              f"Por criticidade: {por_crit}", f"Por categoria: {por_cat}",
              f"Clima (fonte {clima['fonte']}): chuva dia {clima['precipitacao_dia_mm']} mm, "
              f"rajada dia {clima['rajada_dia_kmh']} km/h",
              f"Itens para REVISÃO HUMANA: {len(revisao)}", ""]
    if not top.empty:
        linhas.append("Top 10 prioridades:")
        linhas += [f"- [{r.criticidade}] {r.id_chamado} | {r.categoria.replace('_', ' ')} | {r.bairro}"
                   for r in top.itertuples()]
    msg.set_content("\n".join(linhas))
    corpo = (f"<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'><title>Resumo de zeladoria</title>"
             f"<style>{CSS_ACESSIVEL}</style></head><body><main><h1>Resumo de zeladoria</h1>"
             f"<p>Execução {html.escape(resumo['execucao_id'])}: {resumo['linhas_lidas']} linhas lidas, "
             f"{resumo['processados_novos']} processadas, {resumo['em_quarentena']} em quarentena, "
             f"{len(revisao)} para revisão humana.</p>"
             f"<p>Por criticidade: {html.escape(str(por_crit))}</p><p>Por categoria: {html.escape(str(por_cat))}</p>"
             f"{'<h2>Top 10 prioridades</h2>' + _tabela_html(top) if not top.empty else ''}"
             f"</main></body></html>")
    msg.add_alternative(corpo, subtype="html")
    return msg


def _salvar_em_outbox(msg: EmailMessage, config: Config, rotulo: str) -> str:
    """Caminho seguro: preserva a mensagem em outbox/ como .eml (nada se perde)."""
    pasta = Path(config.output_dir) / "outbox"
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{rotulo}.eml"
    caminho.write_bytes(bytes(msg))
    return str(caminho)


def enviar_email(msg: EmailMessage, config: Config, rotulo: str,
                 dormir: Callable[[float], None] = time.sleep) -> str:
    """
    Envia (ou simula) o e-mail. Retorna: 'enviado' | 'simulado' | 'falha_outbox'.
    * modo simulado: grava .eml em outbox/ (sem rede).
    * modo smtp: STARTTLS/SSL, timeout, retry com espera; em qualquer falha a
      mensagem é salva em outbox/ e a execução continua.
    """
    if config.modo_email != "smtp":
        caminho = _salvar_em_outbox(msg, config, rotulo)
        registrar(logging.INFO, "NOTIFICACAO", "SIMULADO", f"'{rotulo}' gravado em {caminho}")
        return "simulado"

    if not (config.smtp_host and config.smtp_user and config.smtp_password):
        caminho = _salvar_em_outbox(msg, config, rotulo)
        registrar(logging.ERROR, "NOTIFICACAO", "SMTP_NAO_CONFIGURADO",
                  f"credenciais SMTP ausentes no .env; '{rotulo}' preservado em {caminho}")
        return "falha_outbox"

    ultimo_erro = ""
    for tentativa in range(1, config.smtp_retries + 1):
        try:
            if config.smtp_port == 465:
                servidor = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=config.smtp_timeout)
            else:
                servidor = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=config.smtp_timeout)
                servidor.starttls()
            with servidor:
                servidor.login(config.smtp_user, config.smtp_password)
                servidor.send_message(msg)
            registrar(logging.INFO, "NOTIFICACAO", "ENVIADO", f"'{rotulo}' enviado via SMTP (tentativa {tentativa})")
            return "enviado"
        except (smtplib.SMTPException, OSError) as exc:      # OSError cobre timeout/conexão recusada
            ultimo_erro = f"{type(exc).__name__}: {exc}"
            registrar(logging.WARNING, "NOTIFICACAO", "FALHA_SMTP",
                      f"tentativa {tentativa}/{config.smtp_retries} falhou ({ultimo_erro})")
            if tentativa < config.smtp_retries:
                dormir(config.api_backoff * (2 ** (tentativa - 1)))
    caminho = _salvar_em_outbox(msg, config, rotulo)
    registrar(logging.ERROR, "NOTIFICACAO", "FALLBACK_OUTBOX",
              f"SMTP indisponível ({ultimo_erro}); '{rotulo}' preservado em {caminho} para reenvio")
    return "falha_outbox"


# =============================================================================
# 8. RELATÓRIO DE EXECUÇÃO
# =============================================================================

def gravar_relatorios(resumo: Dict[str, Any], df: pd.DataFrame, quarentena: pd.DataFrame,
                      config: Config) -> None:
    """Grava CSV priorizado, CSV de quarentena, JSON e Markdown do resumo."""
    saida = Path(config.output_dir)
    try:
        saida.mkdir(parents=True, exist_ok=True)
        df.to_csv(saida / "chamados_priorizados.csv", index=False, encoding="utf-8")
        quarentena.to_csv(saida / "quarentena_chamados.csv", index=False, encoding="utf-8")
        (saida / "resumo_execucao.json").write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        linhas = [f"# Relatório de execução - {resumo['execucao_id']}", ""] + \
                 [f"- **{k}**: {v}" for k, v in resumo.items()]
        (saida / "resumo_execucao.md").write_text("\n".join(linhas), encoding="utf-8")
        registrar(logging.INFO, "RELATORIO", "OK", f"relatórios gravados em '{saida}/'")
    except OSError as exc:
        registrar(logging.ERROR, "RELATORIO", "FALHA_ESCRITA", f"{type(exc).__name__}: {exc}")


# =============================================================================
# 9. ORQUESTRAÇÃO PONTA A PONTA
# =============================================================================

def executar_pipeline(config: Config, http_get: Callable[..., Any] = requests.get,
                      dormir: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Executa o fluxo completo e devolve o resumo. Cada etapa tem barreira própria."""
    contador = configurar_logging(config.log_file)
    inicio = time.monotonic()
    execucao_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    registrar(logging.INFO, "INICIO", "OK", f"execução {execucao_id} iniciada (modo e-mail: {config.modo_email})")

    # --- Etapa 1: clima (Tema 1) ---
    breaker = CircuitBreaker(config.cb_threshold, config.cb_cooldown)
    clima = coletar_clima(config, breaker, http_get=http_get, dormir=dormir)
    alerta_clima = ha_alerta_climatico(clima)

    # --- Etapa 2: ingestão e validação (Tema 2) ---
    df_bruto, malformadas, leitura_ok = carregar_chamados(config)
    df_validos, quarentena, mascaras = validar_chamados(df_bruto)
    if malformadas:                       # linhas com nº de campos errado também vão à quarentena
        extra = pd.DataFrame([{"id_chamado": "", "data_registro": "", "categoria": "", "descricao": m["linha_bruta"],
                               "latitude": "", "longitude": "", "motivo": "linha_malformada"} for m in malformadas])
        quarentena = pd.concat([quarentena, extra], ignore_index=True)

    # --- Etapa 3: priorização por regras inteligentes ---
    df_prior = priorizar(df_validos, clima, config)

    # --- Etapa 4: persistência idempotente ---
    df_novos = persistir(df_prior, config, execucao_id)

    # --- Etapa 5: notificações (Tema 3) ---
    criticos = df_novos[df_novos["criticidade"] == "CRITICA"] if not df_novos.empty else df_novos
    alertas_disparados = len(criticos) + (1 if alerta_clima else 0)
    status_emails: Dict[str, str] = {}
    try:
        msg_alerta = montar_alerta_emergencia(criticos, clima, alerta_clima, config)
        if msg_alerta is not None:
            status_emails["alerta_emergencia"] = enviar_email(msg_alerta, config, "alerta_emergencia", dormir)
        resumo_parcial = {"execucao_id": execucao_id, "linhas_lidas": len(df_bruto) + len(malformadas),
                          "processados_novos": len(df_novos), "em_quarentena": len(quarentena)}
        status_emails["resumo_zeladoria"] = enviar_email(
            montar_resumo_zeladoria(df_novos, resumo_parcial, clima, config), config, "resumo_zeladoria", dormir)
    except Exception as exc:  # noqa: BLE001 - nenhuma falha de e-mail derruba o robô
        registrar(logging.ERROR, "NOTIFICACAO", "ERRO_INESPERADO", f"{type(exc).__name__}: {exc}")

    # --- Etapa 6: relatório de execução ---
    linhas_lidas = len(df_bruto) + len(malformadas)
    resumo = {
        "execucao_id": execucao_id,
        "csv_lido_com_sucesso": leitura_ok,
        "linhas_lidas": linhas_lidas,
        "validos": len(df_validos),
        "em_quarentena": len(quarentena),
        "taxa_validos_pct": round(100 * len(df_validos) / linhas_lidas, 1) if linhas_lidas else 0.0,
        "processados_novos": len(df_novos),
        "ja_processados_anteriormente": len(df_validos) - len(df_novos),
        "dados_pessoais_mascarados": mascaras,
        "distribuicao_criticidade": df_prior["criticidade"].value_counts().to_dict() if not df_prior.empty else {},
        "revisao_humana": int(df_prior["revisao_humana"].sum()) if not df_prior.empty else 0,
        "alertas_criticos": len(criticos),
        "alerta_climatico": alerta_clima,
        "alertas_disparados": alertas_disparados,
        "status_emails": status_emails,
        "fonte_clima": clima["fonte"],
        "fallback_clima_acionado": clima["fonte"] != "api",
        "logs_warning": contador.contagem["WARNING"],
        "logs_error": contador.contagem["ERROR"],
        "logs_critical": contador.contagem["CRITICAL"],
        "duracao_s": round(time.monotonic() - inicio, 2),
    }
    gravar_relatorios(resumo, df_prior, quarentena, config)
    registrar(logging.INFO, "FIM", "OK",
              f"processados={resumo['processados_novos']} quarentena={resumo['em_quarentena']} "
              f"alertas={resumo['alertas_disparados']} duração={resumo['duracao_s']}s")
    return resumo


def main(argv: Optional[List[str]] = None) -> int:
    """Ponto de entrada. Retorna 0 (ok) ou 1 (falha fatal inesperada, já registrada em CRITICAL)."""
    ap = argparse.ArgumentParser(description="Robô RPA + IA - Cidade Para Todos")
    ap.add_argument("--csv", help="caminho do CSV de chamados")
    ap.add_argument("--output-dir", help="pasta de saída")
    ap.add_argument("--modo-email", choices=["simulado", "smtp"], help="simulado (padrão) ou smtp")
    args = ap.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.csv:
        overrides["csv_path"] = args.csv
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
        overrides["db_path"] = str(Path(args.output_dir) / "cidade_para_todos.db")
    if args.modo_email:
        overrides["modo_email"] = args.modo_email
    config = carregar_config(**overrides)

    try:
        executar_pipeline(config)
        return 0
    except Exception as exc:  # noqa: BLE001 - barreira final
        configurar_logging(config.log_file)
        registrar(logging.CRITICAL, "FATAL", "FALHA_NAO_TRATADA", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
