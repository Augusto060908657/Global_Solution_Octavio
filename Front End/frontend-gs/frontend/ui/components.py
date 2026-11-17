"""Componentes visuais reutilizáveis e acessíveis.

Regras de acessibilidade aplicadas aqui:
- cor nunca é o único sinal: todo card tem ÍCONE + TEXTO do nível (ex.: "🚨 Urgência crítica");
- contraste: texto escuro sobre fundo claro (>= 7:1) e borda lateral grossa colorida;
- role="alert" para situações críticas/erros e role="status" para o resto (leitores de tela);
- todo texto vindo da IA passa por html.escape (evita injeção de HTML).
"""
import html

import streamlit as st

from providers.api_client import ApiError

URGENCY_STYLE = {
    "baixa":   {"icone": "✅", "rotulo": "Urgência baixa",   "bg": "#E8F5E9", "fg": "#0B3D1A", "borda": "#2E7D32"},
    "media":   {"icone": "ℹ️", "rotulo": "Urgência média",   "bg": "#FFF8E1", "fg": "#3D2E00", "borda": "#F9A825"},
    "alta":    {"icone": "⚠️", "rotulo": "Urgência alta",    "bg": "#FFF0E0", "fg": "#4A2000", "borda": "#EF6C00"},
    "critica": {"icone": "🚨", "rotulo": "Urgência crítica", "bg": "#FDE8E8", "fg": "#5F0A0A", "borda": "#C62828"},
}
STATUS_STYLE = {
    "sucesso": {"icone": "✅", "bg": "#E8F5E9", "fg": "#0B3D1A", "borda": "#2E7D32"},
    "info":    {"icone": "💡", "bg": "#E3F2FD", "fg": "#0A2A4D", "borda": "#1565C0"},
    "erro":    {"icone": "⛔", "bg": "#FDE8E8", "fg": "#5F0A0A", "borda": "#C62828"},
}
ALL_STYLES = {**URGENCY_STYLE, **STATUS_STYLE}
URGENCY_COLORS = {"Baixa": "#2E7D32", "Média": "#F9A825", "Alta": "#EF6C00", "Crítica": "#C62828"}


def urgency_text(urgencia: str) -> str:
    """'media' -> 'Média' (usado nos gráficos)."""
    return URGENCY_STYLE[urgencia]["rotulo"].replace("Urgência ", "").capitalize()


def _e(text) -> str:
    return html.escape(str(text if text is not None else ""))


# ------------------------------------------------------------------ CSS global
def inject_css(high_contrast: bool = False, large_font: bool = False) -> None:
    base = """
    .ct-card{border-radius:10px;padding:14px 18px;margin:8px 0 12px 0;line-height:1.5;border-left:8px solid;}
    .ct-card strong.ct-title{font-size:1.1rem;display:block;margin-bottom:4px;}
    .ct-badge{display:inline-block;border-radius:999px;padding:2px 12px;font-weight:700;border:2px solid;font-size:.9rem;}
    .ct-steps{margin:6px 0 0 0;padding-left:22px;}
    *:focus-visible{outline:3px solid #1565C0 !important;outline-offset:2px;}
    """
    if large_font:
        base += ".stApp, .stApp p, .stApp label, .stApp li, .stApp span{font-size:1.15rem !important;}"
    if high_contrast:
        base += """
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"]{background:#000 !important;}
        .stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp div[data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"]{color:#FFF !important;}
        .stApp button{border:2px solid #FFEB3B !important;}
        .ct-card{background:#000 !important;color:#FFF !important;border:3px solid #FFEB3B !important;border-left-width:10px !important;}
        .ct-card *{color:#FFF !important;}
        """
    st.markdown(f"<style>{base}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ componentes
def status_card(nivel: str, titulo: str, corpo: str = "", icone: str | None = None) -> None:
    """Card genérico (reutilizado em urgência, sucesso, erro, avisos, mensagens de API...)."""
    s = ALL_STYLES[nivel]
    role = "alert" if nivel in ("critica", "erro") else "status"
    corpo_html = f"<div>{_e(corpo)}</div>" if corpo else ""
    st.markdown(
        f'<div class="ct-card" role="{role}" style="background:{s["bg"]};color:{s["fg"]};border-color:{s["borda"]};">'
        f'<strong class="ct-title">{_e(icone or s["icone"])} {_e(titulo)}</strong>{corpo_html}</div>',
        unsafe_allow_html=True)


def urgency_badge(urgencia: str) -> str:
    """HTML inline do selo de urgência (ícone + texto + cor)."""
    s = URGENCY_STYLE[urgencia]
    return (f'<span class="ct-badge" style="background:{s["bg"]};color:{s["fg"]};border-color:{s["borda"]};">'
            f'{s["icone"]} {s["rotulo"]}</span>')


def empty_state(titulo: str, texto: str, icone: str = "📭") -> None:
    status_card("info", titulo, texto, icone)


def error_message(err: ApiError) -> None:
    """Mostra ApiError ao cidadão: mensagem clara, sem termos técnicos."""
    status_card("erro", err.mensagem, err.sugestao)


def flash_message(flash: dict | None) -> None:
    if flash:
        nivel = flash["nivel"] if flash["nivel"] in ALL_STYLES else "info"
        status_card(nivel, flash["titulo"], flash.get("texto", ""))


def result_card(item: dict, compact: bool = False) -> None:
    """Resultado da IA em formato útil ao cidadão (usado no formulário E no histórico)."""
    if item.get("alerta_seguranca") or item.get("risco_imediato"):
        fones = " • ".join(item.get("telefones") or [])
        status_card("critica", "Possível situação de risco — ligue agora", fones)

    st.markdown(
        f'{urgency_badge(item["urgencia"])} &nbsp; <strong>{_e(item["categoria_label"])}</strong> '
        f'&nbsp;·&nbsp; Protocolo <code>{_e(item["protocolo"])}</code>', unsafe_allow_html=True)
    st.markdown(f"**Resumo:** {item['resumo']}")
    if item.get("justificativa_urgencia") and not compact:
        st.caption(f"Por que esta urgência? {item['justificativa_urgencia']}")

    col1, col2 = st.columns([3, 2])
    with col1:
        passos = "".join(f"<li>{_e(p)}</li>" for p in item["orientacoes"])
        status_card("info", "O que fazer agora", icone="🧭")
        st.markdown(f'<ol class="ct-steps">{passos}</ol>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**🏛️ Quem resolve:** {item['orgao']}")
        st.markdown(f"**⏱️ Prazo estimado:** {item['prazo']}")
        if item.get("bairro"):
            st.markdown(f"**📍 Local:** {item['bairro']}")
        if item.get("telefones") and not item.get("alerta_seguranca"):
            st.caption("📞 " + " • ".join(item["telefones"]))

    if not compact:
        confianca = item["confianca"]
        st.progress(confianca, text=f"Confiança da IA: {confianca:.0%}")
        if confianca < 0.5:
            status_card("media", "A IA não tem muita certeza",
                        "Confira com atenção e corrija a categoria se necessário.", "🤔")
        if item.get("mascarado"):
            st.caption("🔒 Dados pessoais (CPF, telefone, e-mail) foram ocultados antes da análise.")
        st.caption(f"Orientação gerada por IA ({item['modelo']}, {item['latencia_ms'] / 1000:.1f}s). "
                   "Não substitui o atendimento oficial.")
