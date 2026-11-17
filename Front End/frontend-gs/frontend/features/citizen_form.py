"""Tela 1 — Nova solicitação: formulário validado -> IA -> resultado -> confirmação humana."""
import re

import streamlit as st

from providers.api_client import ApiError
from state import session_manager as sm
from ui import components as ui

EXEMPLOS = {
    "Poste apagado": "A lâmpada do poste da Rua das Acácias, em frente ao número 120, está queimada há duas "
                     "semanas e a rua fica muito escura à noite.",
    "Entulho na calçada": "Deixaram um monte de entulho de obra na calçada da escola do bairro e as crianças "
                          "precisam andar no meio da rua.",
    "Cheiro de gás": "Estou sentindo um cheiro forte de gás no corredor do meu prédio desde hoje cedo.",
}
DECISOES = {"Sim, a classificação está correta": "confirmado",
            "Não, quero corrigir a categoria": "corrigido",
            "Descartar (não registrar)": "descartado"}


def validar(relato: str, cep: str, aceite: bool) -> list[str]:
    erros = []
    relato = (relato or "").strip()
    if not relato:
        erros.append("Descreva o problema no campo “O que está acontecendo?”.")
    elif len(relato) < 20:
        erros.append("Conte um pouco mais: o relato precisa ter pelo menos 20 caracteres.")
    elif len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", relato)) < 3:
        erros.append("O relato precisa ter pelo menos 3 palavras.")
    if cep and len(re.sub(r"\D", "", cep)) != 8:
        erros.append("O CEP deve ter 8 números (ex.: 01310-100) ou ficar em branco.")
    if not aceite:
        erros.append("Marque a confirmação de que você não incluiu dados pessoais sensíveis.")
    return erros


def _usar_exemplo(texto: str, nonce: int) -> None:
    st.session_state[f"relato_{nonce}"] = texto


def _confirmar(protocolo: str) -> None:
    """Callback do botão: lê os widgets pelo key e registra a decisão."""
    decisao = DECISOES[st.session_state[f"decisao_{protocolo}"]]
    pending = st.session_state.pending
    categoria = st.session_state.get(f"cat_{protocolo}") if decisao == "corrigido" else pending["categoria"]
    aval = {"👍 Foi útil": "util", "👎 Está incorreta ou confusa": "incorreta"}.get(
        st.session_state.get(f"aval_{protocolo}"))
    sm.register_request(decisao, categoria or pending["categoria"], aval,
                        st.session_state.get(f"coment_{protocolo}", ""))


def _render_pendente(cats: dict) -> None:
    item = st.session_state.pending
    proto = item["protocolo"]
    st.divider()
    st.subheader("2. Resultado da análise")
    ui.result_card(item)

    st.subheader("3. Confira antes de registrar")
    st.caption("A IA pode errar. Você decide se o resultado está certo antes de registrar o pedido.")
    with st.container(border=True):
        escolha = st.radio("A classificação está correta?", list(DECISOES), key=f"decisao_{proto}")
        if DECISOES[escolha] == "corrigido":
            ids = list(cats)
            st.selectbox("Qual é a categoria correta?", ids, index=ids.index(item["categoria"])
                         if item["categoria"] in ids else 0, format_func=lambda k: cats[k], key=f"cat_{proto}")
        st.radio("A orientação foi útil?", ["👍 Foi útil", "👎 Está incorreta ou confusa"], index=None,
                 horizontal=True, key=f"aval_{proto}")
        st.text_input("Comentário (opcional)", max_chars=200, key=f"coment_{proto}",
                      help="Não escreva dados pessoais aqui.")
        rotulo = "🗑️ Descartar" if DECISOES[escolha] == "descartado" else "✅ Confirmar e registrar solicitação"
        st.button(rotulo, type="primary", on_click=_confirmar, args=(proto,), key=f"btn_{proto}")


def render() -> None:
    nonce = st.session_state.form_nonce
    cats = sm.cached_categories()

    ui.flash_message(st.session_state.pop("flash", None))
    st.subheader("1. Conte o que está acontecendo")
    st.caption("Descreva o problema com suas palavras. Campos com * são obrigatórios.")

    st.markdown("**Sem ideias? Teste com um exemplo:**")
    cols = st.columns(len(EXEMPLOS))
    for col, (nome, texto) in zip(cols, EXEMPLOS.items()):
        col.button(nome, on_click=_usar_exemplo, args=(texto, nonce), key=f"ex_{nome}_{nonce}")

    with st.form(f"form_{nonce}"):
        relato = st.text_area("O que está acontecendo? *", key=f"relato_{nonce}", max_chars=1000, height=150,
                              placeholder="Ex.: Há um buraco grande na Rua X, perto do número 50...",
                              help="De 20 a 1000 caracteres. Diga onde fica e há quanto tempo acontece.")
        c1, c2, c3 = st.columns(3)
        bairro = c1.text_input("Bairro", max_chars=80, key=f"bairro_{nonce}")
        cep = c2.text_input("CEP (opcional)", max_chars=9, key=f"cep_{nonce}",
                            help="Usamos o CEP para encontrar o endereço em dados públicos (ViaCEP).")
        sugerida = c3.selectbox("Tipo de problema (opcional)", ["", *cats],
                                format_func=lambda k: cats.get(k, "A IA decide"), key=f"sug_{nonce}")
        aceite = st.checkbox("Confirmo que não incluí CPF, telefone ou outros dados pessoais sensíveis. *",
                             key=f"aceite_{nonce}")
        enviar = st.form_submit_button("🔍 Analisar meu relato", type="primary")

    if enviar:
        erros = validar(relato, cep, aceite)
        if erros:
            for e in erros:
                ui.status_card("erro", e, icone="✏️")
        else:
            try:
                with st.status("Analisando seu relato…", expanded=True) as status:
                    st.write("✔️ Relato validado")
                    st.write("🤖 A IA local está lendo e classificando (pode levar até 1 minuto na 1ª vez)…")
                    resp = sm.cached_analyze(relato.strip(), (bairro or "").strip(), re.sub(r"\D", "", cep or ""),
                                             sugerida or "")
                    st.write("✔️ Organizando orientações")
                    status.update(label="Análise concluída", state="complete", expanded=False)
                st.session_state.pending = sm.flatten_analysis(resp, relato.strip(), (bairro or "").strip())
            except ApiError as err:
                st.session_state.pending = None
                ui.error_message(err)

    if st.session_state.pending:
        _render_pendente(cats)
