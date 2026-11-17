"""Tela 2 — Painel: gráficos Plotly interativos (hover, zoom) alimentados pelo histórico + 3 filtros."""
import plotly.express as px
import streamlit as st

from state import session_manager as sm
from ui import components as ui


def _fmt_urg(u: str) -> str:
    s = ui.URGENCY_STYLE[u]
    return f"{s['icone']} {s['rotulo']}"


def _estilizar(fig, altura: int = 420):
    fig.update_layout(height=altura, font=dict(size=14), margin=dict(t=60, b=20, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0))
    return fig


def render() -> None:
    st.subheader("Painel de solicitações")
    ui.flash_message(st.session_state.pop("flash", None))
    history = st.session_state.history
    if not history:
        ui.empty_state("Ainda não há solicitações",
                       "Registre uma solicitação na aba “Nova solicitação” ou carregue dados de exemplo.")
        st.button("📥 Carregar dados de exemplo", on_click=sm.seed_demo_data, key="dash_seed")
        return

    df = sm.history_dataframe(history)
    dmin, dmax = df["data"].min(), df["data"].max()

    with st.container(border=True):
        st.markdown("**Filtros** — deixe vazio para ver tudo")
        f1, f2, f3, f4 = st.columns(4)
        cats = f1.multiselect("Categoria", sorted(df["categoria_label"].unique()), key="dash_cat",
                              placeholder="Todas as categorias")
        urgs = f2.multiselect("Urgência", sm.URGENCY_ORDER, format_func=_fmt_urg, key="dash_urg",
                              placeholder="Todas as urgências")
        stat = f3.multiselect("Situação", sorted(df["status"].unique()), key="dash_status",
                              placeholder="Todas as situações")
        periodo = f4.date_input("Período", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="dash_periodo",
                                format="DD/MM/YYYY")

    fdf = df
    if cats:
        fdf = fdf[fdf["categoria_label"].isin(cats)]
    if urgs:
        fdf = fdf[fdf["urgencia"].isin(urgs)]
    if stat:
        fdf = fdf[fdf["status"].isin(stat)]
    if isinstance(periodo, (tuple, list)) and len(periodo) == 2:
        fdf = fdf[(fdf["data"] >= periodo[0]) & (fdf["data"] <= periodo[1])]

    if fdf.empty:
        ui.empty_state("Nenhuma solicitação com esses filtros", "Tente remover algum filtro.", "🔎")
        return
    if df["demo"].any():
        st.caption("ℹ️ O painel inclui dados de exemplo (fictícios).")

    graves = int(fdf["urgencia"].isin(["alta", "critica"]).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Solicitações", len(fdf))
    m2.metric("Urgência alta ou crítica", graves, f"{graves / len(fdf):.0%} do total", delta_color="off")
    m3.metric("Confirmadas pelo cidadão", int((fdf["status"] != "Descartada").sum()))
    m4.metric("Categoria mais comum", fdf["categoria_label"].mode().iat[0])

    fdf = fdf.assign(Urgência=fdf["urgencia"].astype(str).map(ui.urgency_text))
    ordem = [ui.urgency_text(u) for u in sm.URGENCY_ORDER]
    aba1, aba2, aba3 = st.tabs(["📊 Por categoria", "📈 Linha do tempo", "🧩 Situação dos pedidos"])

    with aba1:
        agg = fdf.groupby(["categoria_label", "Urgência"], observed=True).size().reset_index(name="Quantidade")
        fig = px.bar(agg, x="categoria_label", y="Quantidade", color="Urgência", pattern_shape="Urgência", text="Quantidade",
                     color_discrete_map=ui.URGENCY_COLORS, category_orders={"Urgência": ordem},
                     labels={"categoria_label": "Categoria"}, title="Solicitações por categoria e urgência")
        st.plotly_chart(_estilizar(fig), key="fig_categoria")
        st.caption("Cada urgência tem cor E padrão de preenchimento diferentes. Passe o mouse para ver detalhes; "
                   "arraste para ampliar.")
        with st.expander("Ver os dados deste gráfico em tabela"):
            st.dataframe(agg, hide_index=True)

    with aba2:
        fig = px.scatter(fdf, x="criado_em", y="Urgência", color="Urgência", symbol="Urgência",
                         color_discrete_map=ui.URGENCY_COLORS, category_orders={"Urgência": ordem},
                         hover_data={"protocolo": True, "categoria_label": True, "bairro": True, "criado_em": "|%d/%m %H:%M"},
                         labels={"criado_em": "Data e hora", "categoria_label": "Categoria"},
                         title="Quando cada solicitação foi feita")
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="#000")))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(_estilizar(fig, 500), key="fig_timeline")
        st.caption("Use a barra abaixo do gráfico para dar zoom em um período.")

    with aba3:
        cont = fdf["status"].value_counts().reset_index()
        cont.columns = ["Situação", "Quantidade"]
        fig = px.pie(cont, names="Situação", values="Quantidade", hole=0.45, title="Situação das solicitações",
                     color_discrete_sequence=["#1565C0", "#6A1B9A", "#616161"])
        fig.update_traces(textinfo="label+value+percent", textfont_size=14)
        st.plotly_chart(_estilizar(fig), key="fig_status")
        aval = fdf["avaliacao"].value_counts()
        st.markdown(f"**Avaliações da IA:** 👍 útil: {int(aval.get('util', 0))} · "
                    f"👎 incorreta: {int(aval.get('incorreta', 0))}")
