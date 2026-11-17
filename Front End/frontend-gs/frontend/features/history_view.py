"""Tela 3 — Histórico de solicitações do cidadão (reutiliza ui.result_card)."""
from datetime import datetime

import streamlit as st

from state import session_manager as sm
from ui import components as ui


def render() -> None:
    st.subheader("Meu histórico")
    history = st.session_state.history
    if not history:
        ui.empty_state("Você ainda não registrou solicitações",
                       "Quando você confirmar uma solicitação, ela aparece aqui com o protocolo.")
        return

    df = sm.history_dataframe(history)
    c1, c2 = st.columns([2, 2])
    busca = c1.text_input("Buscar por palavra ou protocolo", key="hist_busca", max_chars=60)
    status = c2.multiselect("Situação", sorted(df["status"].unique()), key="hist_status", placeholder="Todas")

    itens = sorted(history, key=lambda h: h["criado_em"], reverse=True)
    if busca:
        termo = busca.lower()
        itens = [h for h in itens if termo in h["relato"].lower() or termo in h["protocolo"].lower()
                 or termo in h["categoria_label"].lower()]
    if status:
        itens = [h for h in itens if h["status"] in status]

    st.caption(f"{len(itens)} de {len(history)} solicitações")
    if not itens:
        ui.empty_state("Nada encontrado", "Tente outra palavra ou limpe os filtros.", "🔎")

    for h in itens:
        quando = datetime.fromisoformat(h["criado_em"]).strftime("%d/%m/%Y %H:%M")
        icone = ui.URGENCY_STYLE[h["urgencia"]]["icone"]
        with st.expander(f"{icone} {h['categoria_label']} — {quando} — {h['status']}"):
            st.markdown(f"**Seu relato:** {h['relato']}")
            ui.result_card(h, compact=True)       # mesmo componente da tela 1, com outro parâmetro
            if h.get("avaliacao"):
                st.caption("Sua avaliação: " + ("👍 útil" if h["avaliacao"] == "util" else "👎 incorreta"))

    st.divider()
    csv = df.drop(columns=["relato", "orientacoes", "telefones"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Baixar histórico (CSV, sem o texto do relato)", csv, "historico_citizentech.csv", "text/csv")
    with st.expander("Apagar histórico"):
        ok = st.checkbox("Entendo que isto apaga todas as solicitações desta sessão", key="hist_confirma_apagar")
        st.button("🗑️ Apagar tudo", disabled=not ok, on_click=sm.clear_history, key="hist_apagar")
