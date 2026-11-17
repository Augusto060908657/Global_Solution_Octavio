"""Estado centralizado (st.session_state) + cache (@st.cache_resource / @st.cache_data)."""
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from providers.api_client import ApiClient, ApiError

URGENCY_ORDER = ["baixa", "media", "alta", "critica"]
STATUS_LABELS = {"confirmado": "Confirmada", "corrigido": "Corrigida pelo cidadão", "descartado": "Descartada"}
FALLBACK_CATEGORIES = {
    "iluminacao_publica": "Iluminação pública", "limpeza_urbana_entulho": "Limpeza urbana e entulho",
    "buraco_pavimentacao": "Buracos e pavimentação", "saude": "Saúde", "seguranca": "Segurança",
    "saneamento_agua_esgoto": "Água, esgoto e drenagem", "arvores_areas_verdes": "Árvores e áreas verdes",
    "transporte_mobilidade": "Transporte e mobilidade", "acessibilidade": "Acessibilidade",
    "outros": "Outros serviços públicos",
}

_DEFAULTS = {"history": [], "pending": None, "form_nonce": 0, "flash": None,
             "high_contrast": False, "large_font": False}


# ------------------------------------------------------------------ cache
@st.cache_resource
def get_client() -> ApiClient:
    """Um único cliente HTTP (e sessão TCP) para todos os reruns e usuários do processo."""
    return ApiClient(os.getenv("API_BASE_URL", "http://localhost:8000"))


@st.cache_data(ttl=15, show_spinner=False)
def cached_health() -> dict:
    try:
        return get_client().health()
    except ApiError as err:
        return {"status": "offline", "mensagem": err.mensagem, "ollama": False, "modelo": "—"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_categories() -> dict:
    try:
        return get_client().categorias()
    except ApiError:
        return FALLBACK_CATEGORIES


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def cached_analyze(relato: str, bairro: str, cep: str, categoria_sugerida: str) -> dict:
    """Mesmo relato => mesma resposta, sem nova chamada ao LLM. Erros NÃO são cacheados."""
    payload = {"relato": relato, "bairro": bairro or None, "cep": cep or None,
               "categoria_sugerida": categoria_sugerida or None}
    return get_client().analyze(payload)


@st.cache_data(show_spinner=False)
def history_dataframe(history: list[dict]) -> pd.DataFrame:
    """Histórico -> DataFrame (recalculado só quando o histórico muda)."""
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df["criado_em"] = pd.to_datetime(df["criado_em"])
    df["data"] = df["criado_em"].dt.date
    df["urgencia"] = pd.Categorical(df["urgencia"], categories=URGENCY_ORDER, ordered=True)
    return df


# ------------------------------------------------------------------ estado
def init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value


def flatten_analysis(resp: dict, relato: str, bairro: str) -> dict:
    """Resposta aninhada da API -> registro plano usado por toda a interface."""
    a = resp["analise"]
    end = resp.get("endereco") or {}
    return {
        "protocolo": resp["protocolo"], "criado_em": resp["criado_em"], "relato": relato,
        "bairro": bairro or end.get("bairro") or "Não informado",
        "categoria": a["categoria"], "categoria_label": resp["categoria_label"],
        "urgencia": a["urgencia"], "resumo": a["resumo"],
        "justificativa_urgencia": a.get("justificativa_urgencia", ""),
        "orgao": a["orgao_responsavel"], "orientacoes": a["orientacoes"], "prazo": a["prazo_estimado"],
        "risco_imediato": bool(a.get("risco_imediato")), "confianca": float(a["confianca"]),
        "telefones": resp.get("telefones_emergencia", []), "alerta_seguranca": resp.get("alerta_seguranca", False),
        "mascarado": resp.get("dados_pessoais_mascarados", False),
        "cidade": end.get("cidade"), "modelo": resp["modelo"], "latencia_ms": resp["latencia_ms"],
        "status": "Aguardando confirmação", "avaliacao": None, "comentario": "", "demo": False,
        "categoria_final": a["categoria"],
    }


def set_flash(nivel: str, titulo: str, texto: str = "") -> None:
    st.session_state.flash = {"nivel": nivel, "titulo": titulo, "texto": texto}


def register_request(decisao: str, categoria_final: str, avaliacao: str | None, comentario: str) -> None:
    """Human-in-the-loop: grava a decisão do cidadão no histórico e no back-end."""
    item = dict(st.session_state.pending)
    cats = cached_categories()
    item.update(status=STATUS_LABELS[decisao], avaliacao=avaliacao, comentario=comentario.strip(),
                categoria_final=categoria_final)
    if decisao == "corrigido":
        item["categoria"], item["categoria_label"] = categoria_final, cats.get(categoria_final, categoria_final)
    try:
        get_client().send_feedback({"protocolo": item["protocolo"], "decisao": decisao,
                                    "categoria_final": categoria_final, "avaliacao": avaliacao,
                                    "comentario": item["comentario"] or None})
        sincronizado = True
    except ApiError:
        sincronizado = False
    history = [h for h in st.session_state.history if h["protocolo"] != item["protocolo"]]
    st.session_state.history = history + [item]
    st.session_state.pending = None
    st.session_state.form_nonce += 1                # limpa o formulário
    if decisao == "descartado":
        set_flash("info", "Solicitação descartada", "Nada foi registrado como pedido. Você pode escrever outro relato.")
    elif sincronizado:
        set_flash("sucesso", f"Solicitação registrada — protocolo {item['protocolo']}",
                  "Guarde este número para acompanhar seu pedido.")
    else:
        set_flash("media", f"Registrada neste aparelho — protocolo {item['protocolo']}",
                  "Não conseguimos enviar sua avaliação ao servidor agora, mas seu histórico foi salvo aqui.")


def clear_history() -> None:
    st.session_state.history = []
    set_flash("info", "Histórico apagado", "Todas as solicitações desta sessão foram removidas.")


# ------------------------------------------------------------------ dados de exemplo (para demonstração)
_DEMO = [
    ("iluminacao_publica", "Iluminação pública", "media", "Prefeitura – iluminação pública",
     "Poste apagado na rua há duas semanas."),
    ("limpeza_urbana_entulho", "Limpeza urbana e entulho", "alta", "Prefeitura – zeladoria",
     "Entulho bloqueando a calçada de uma escola."),
    ("buraco_pavimentacao", "Buracos e pavimentação", "media", "Prefeitura – conservação de vias",
     "Buraco grande na avenida principal."),
    ("saneamento_agua_esgoto", "Água, esgoto e drenagem", "alta", "Companhia de saneamento",
     "Esgoto a céu aberto próximo a casas."),
    ("saude", "Saúde", "alta", "Secretaria Municipal de Saúde / UBS", "Falta de atendimento na UBS do bairro."),
    ("arvores_areas_verdes", "Árvores e áreas verdes", "baixa", "Prefeitura – áreas verdes",
     "Árvore com galhos sobre a fiação."),
    ("seguranca", "Segurança", "critica", "Polícia Militar / Guarda Civil", "Ameaça em andamento na rua."),
    ("acessibilidade", "Acessibilidade", "baixa", "Prefeitura – acessibilidade", "Calçada sem rampa para cadeirantes."),
]
_BAIRROS = ["Centro", "Vila Nova", "Jardim das Flores", "Parque Industrial", "Zona Leste", "Bela Vista"]


def seed_demo_data() -> None:
    """Gera 30 registros FICTÍCIOS (marcados como demo) para demonstrar o painel."""
    rng = random.Random(42)
    agora = datetime.now()
    demo = []
    for i in range(30):
        cat, label, urg, orgao, resumo = rng.choice(_DEMO)
        when = agora - timedelta(days=rng.randint(0, 29), hours=rng.randint(0, 23))
        demo.append({
            "protocolo": f"CT-DEMO-{1000 + i}", "criado_em": when.isoformat(timespec="seconds"),
            "relato": resumo, "bairro": rng.choice(_BAIRROS), "categoria": cat, "categoria_label": label,
            "urgencia": urg, "resumo": resumo, "justificativa_urgencia": "Exemplo para demonstração.",
            "orgao": orgao, "orientacoes": ["Registre o pedido na central 156.", "Guarde o protocolo."],
            "prazo": "5 a 15 dias úteis", "risco_imediato": urg == "critica",
            "confianca": round(rng.uniform(0.55, 0.95), 2), "telefones": [], "alerta_seguranca": False,
            "mascarado": False, "cidade": None, "modelo": "demo", "latencia_ms": rng.randint(2000, 9000),
            "status": rng.choices(list(STATUS_LABELS.values()), weights=[6, 3, 1])[0],
            "avaliacao": rng.choice(["util", "util", "incorreta", None]), "comentario": "",
            "demo": True, "categoria_final": cat})
    reais = [h for h in st.session_state.history if not h.get("demo")]
    st.session_state.history = reais + demo
    set_flash("info", "Dados de exemplo carregados", "São 30 solicitações fictícias, só para demonstração.")
