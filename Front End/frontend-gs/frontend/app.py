"""Ponto de entrada Streamlit — CitizenTech. Orquestra sidebar e tabs; a lógica vive em features/."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import streamlit as st  # noqa: E402

st.set_page_config(page_title="CitizenTech", page_icon="🏙️", layout="wide")

from features import Dashboard, citizen_form, history_view  # noqa: E402
from state import session_manager as sm  # noqa: E402
from ui import components as ui  # noqa: E402

sm.init_state()
ui.inject_css(st.session_state.high_contrast, st.session_state.large_font)

with st.sidebar:
    st.title("🏙️ CitizenTech")
    st.caption("Portal inteligente de atendimento e orientação ao cidadão")

    st.markdown("### ♿ Acessibilidade")
    st.toggle("Alto contraste", key="high_contrast")
    st.toggle("Letras maiores", key="large_font")

    st.markdown("### 🔌 Estado do serviço")
    saude = sm.cached_health()
    if saude.get("status") == "ok":
        ui.status_card("sucesso", "Serviço funcionando", f"IA local: {saude['modelo']}")
    elif saude.get("status") == "degradado":
        ui.status_card("media", "IA indisponível", "Você pode preencher, mas a análise pode falhar.", "⚠️")
    else:
        ui.status_card("erro", "Serviço fora do ar", "Tente novamente em alguns minutos.")
    st.button("🔄 Verificar de novo", on_click=sm.cached_health.clear, key="side_health")

    st.markdown("### 🧭 Como funciona")
    st.markdown("1. Você conta o problema\n2. A IA sugere categoria, urgência e passos\n"
                "3. **Você confere** e registra\n4. Acompanhe no painel")

    st.markdown("### 🎯 ODS")
    st.markdown("**11** – Cidades e Comunidades Sustentáveis  \n**16** – Paz, Justiça e Instituições Eficazes")
    st.divider()
    st.button("📥 Carregar dados de exemplo", on_click=sm.seed_demo_data, key="side_seed")

st.title("Fale com a sua cidade, sem complicação")
st.caption("Em emergência, ligue: 190 Polícia · 192 SAMU · 193 Bombeiros · 199 Defesa Civil.")

tab1, tab2, tab3 = st.tabs(["📝 Nova solicitação", "📊 Painel", "🗂️ Meu histórico"])
with tab1:
    citizen_form.render()
with tab2:
    Dashboard.render()
with tab3:
    history_view.render()
