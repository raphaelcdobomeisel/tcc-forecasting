"""
app/streamlit_app.py
--------------------
Dashboard interativo do TCC — Previsão de Vendas com Prophet + regressor Promo.

Cenários what-if de promoção:
  - Padrão histórico  : repete a taxa de promo por dia da semana
  - Sem promoção      : assume Promo=0 em todos os dias futuros
  - Com promoção total: assume Promo=1 em todos os dias futuros

Rodar com:
    streamlit run app/streamlit_app.py
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_processed
from preprocessing import clean, get_store_series
from model import build_model, predict, save_model
from evaluate import compute_metrics, train_test_split_temporal, metrics_report

PROCESSED_DIR = ROOT / "data" / "processed"

# ── Configuração da página ─────────────────────────────────────
st.set_page_config(
    page_title="TCC — Previsão de Vendas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cache ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando e limpando dados...")
def get_clean_data():
    df = load_processed("rossmann_merged.csv")
    return clean(df)


@st.cache_resource(show_spinner="Treinando modelo Prophet + Promo...")
def get_or_train_model(store_id: int, n_rows: int):
    model_path = PROCESSED_DIR / f"prophet_store_{store_id}.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    df      = get_clean_data()
    series  = get_store_series(df, store_id)
    train_df, _ = train_test_split_temporal(series, test_days=42)
    model   = build_model()
    model.fit(train_df)
    save_model(model, store_id)
    return model


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configurações")

    data_path = PROCESSED_DIR / "rossmann_merged.csv"
    if not data_path.exists():
        st.error(
            "**Dados não encontrados.**\n\n"
            "Execute:\n```\npython src/data_loader.py\n```"
        )
        st.stop()

    df_all     = get_clean_data()
    all_stores = sorted(df_all["Store"].unique())

    store_id      = st.selectbox("🏪 Loja", all_stores, index=0)
    forecast_days = st.slider("🔮 Dias de previsão", 7, 90, 42, 7)
    test_days     = st.slider("🧪 Dias de teste (hold-out)", 14, 90, 42, 7)

    st.markdown("---")
    st.markdown("### 🎭 Cenário de Promoção (futuro)")
    promo_scenario = st.radio(
        "Como simular Promo nos dias futuros?",
        options=["historical_pattern", "all_promo", "no_promo"],
        format_func=lambda x: {
            "historical_pattern": "📊 Padrão histórico (repetir taxa passada)",
            "all_promo":          "🟢 Promoção em todos os dias",
            "no_promo":           "🔴 Sem promoção",
        }[x],
        index=0,
    )
    st.info(
        "💡 **What-if:** mude o cenário e veja como a previsão muda. "
        "O regressor Promo é o principal driver — cortou o MAPE de 15% para 7.5%."
    )

    st.markdown("---")
    st.markdown("**Modelo:** Prophet + Promo  \n**Dataset:** Rossmann Store Sales  \n**TCC:** Ciência de Dados")

# ── Título ─────────────────────────────────────────────────────
scenario_label = {"historical_pattern": "padrão histórico", "all_promo": "com promoção", "no_promo": "sem promoção"}
st.title("📈 Previsão de Vendas — Rossmann Stores")
st.markdown(f"Loja **{store_id}** · Modelo **Prophet + Promo** · {forecast_days} dias · cenário: *{scenario_label[promo_scenario]}*")
st.markdown("---")

# ── Dados e modelo ─────────────────────────────────────────────
series   = get_store_series(df_all, store_id)          # ds, y, Promo
train_df, test_df = train_test_split_temporal(series, test_days=test_days)
model    = get_or_train_model(store_id, n_rows=len(train_df))
forecast = predict(model, historical=series, periods=forecast_days,
                   promo_scenario=promo_scenario)

# Métricas no hold-out (sempre com padrão histórico para consistência)
fc_eval     = predict(model, historical=series, periods=test_days,
                      promo_scenario="historical_pattern")
test_merged = test_df.merge(fc_eval[["ds", "yhat"]], on="ds", how="inner")
metrics     = compute_metrics(test_merged["y"], test_merged["yhat"])

# ── KPIs ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Observações", f"{len(series):,}")
c2.metric("📐 MAE",  f"€{metrics['MAE']:,.0f}")
c3.metric("📐 RMSE", f"€{metrics['RMSE']:,.0f}")
c4.metric("📐 MAPE", f"{metrics['MAPE (%)']:.1f}%",
          delta="-7.9pp vs sem Promo", delta_color="inverse")
st.markdown("---")

# ── Abas ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Previsão", "📊 EDA", "🔬 Componentes", "📋 Avaliação"])

# ── TAB 1: PREVISÃO ────────────────────────────────────────────
with tab1:
    st.subheader("Histórico + Previsão")

    future_fc = forecast[forecast["ds"] > train_df["ds"].max()]
    fig = go.Figure()

    # Histórico treino
    fig.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"],
        mode="lines", name="Histórico (treino)",
        line=dict(color="#1f77b4", width=1.2)))

    # Real teste
    if len(test_df):
        fig.add_trace(go.Scatter(x=test_df["ds"], y=test_df["y"],
            mode="lines+markers", name="Real (teste)",
            line=dict(color="#ff7f0e", width=2), marker=dict(size=4)))

    # Intervalo de confiança
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(44,160,44,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo 80%"))

    # Previsão
    fig.add_trace(go.Scatter(x=future_fc["ds"], y=future_fc["yhat"],
        mode="lines", name="Previsão",
        line=dict(color="#2ca02c", width=2.5, dash="dash")))

    # Dias de promoção futura destacados
    promo_days = future_fc[future_fc["Promo"] == 1]
    if len(promo_days):
        fig.add_trace(go.Scatter(x=promo_days["ds"], y=promo_days["yhat"],
            mode="markers", name="Dias com Promoção",
            marker=dict(color="gold", size=7, symbol="star")))

    fig.add_vline(x=train_df["ds"].max(), line_dash="dot", line_color="gray",
                  annotation_text="início previsão")
    fig.update_layout(
        xaxis_title="Data", yaxis_title="Vendas (€)",
        hovermode="x unified", height=460, template="plotly_white",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela com os próximos dias e flag de promo
    st.markdown("#### Próximas previsões")
    nxt = forecast[forecast["ds"] > train_df["ds"].max()].head(forecast_days).copy()
    nxt_show = nxt[["ds", "yhat", "yhat_lower", "yhat_upper", "Promo"]].copy()
    nxt_show.columns = ["Data", "Previsão (€)", "Mín 80%", "Máx 80%", "Promo?"]
    nxt_show["Data"] = nxt_show["Data"].dt.strftime("%d/%m/%Y")
    for col in ["Previsão (€)", "Mín 80%", "Máx 80%"]:
        nxt_show[col] = nxt_show[col].map("€{:,.0f}".format)
    nxt_show["Promo?"] = nxt_show["Promo?"].map({1: "✅ Sim", 0: "—"})
    st.dataframe(nxt_show, use_container_width=True, hide_index=True)

    # Comparação de cenários de Promo
    st.markdown("#### 📊 Comparação de cenários de promoção")
    fc_all  = predict(model, historical=series, periods=forecast_days, promo_scenario="all_promo")
    fc_none = predict(model, historical=series, periods=forecast_days, promo_scenario="no_promo")
    fc_hist = predict(model, historical=series, periods=forecast_days, promo_scenario="historical_pattern")

    future_all  = fc_all[fc_all["ds"] > train_df["ds"].max()]
    future_none = fc_none[fc_none["ds"] > train_df["ds"].max()]
    future_hist = fc_hist[fc_hist["ds"] > train_df["ds"].max()]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=future_all["ds"],  y=future_all["yhat"],
        mode="lines", name="🟢 Com Promoção total", line=dict(color="green", width=2)))
    fig_comp.add_trace(go.Scatter(x=future_hist["ds"], y=future_hist["yhat"],
        mode="lines", name="📊 Padrão histórico",   line=dict(color="orange", width=2, dash="dot")))
    fig_comp.add_trace(go.Scatter(x=future_none["ds"], y=future_none["yhat"],
        mode="lines", name="🔴 Sem Promoção",        line=dict(color="red", width=2, dash="dash")))
    fig_comp.update_layout(title="Impacto da Promoção na Previsão — 3 Cenários",
        xaxis_title="Data", yaxis_title="Vendas (€)",
        hovermode="x unified", height=380, template="plotly_white",
        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Uplift estimado
    uplift = (future_all["yhat"].mean() / future_none["yhat"].mean() - 1) * 100
    st.success(f"📈 Uplift estimado da promoção: **+{uplift:.1f}%** nas vendas médias previstas.")


# ── TAB 2: EDA ─────────────────────────────────────────────────
with tab2:
    st.subheader("Análise Exploratória")
    store_full = df_all[df_all["Store"] == store_id].copy()

    col_a, col_b = st.columns(2)
    with col_a:
        dow_map = {1:"Seg",2:"Ter",3:"Qua",4:"Qui",5:"Sex",6:"Sáb",7:"Dom"}
        dow = store_full.groupby("DayOfWeek")["Sales"].mean().reset_index()
        dow["DayOfWeek"] = dow["DayOfWeek"].map(dow_map)
        fig_dow = px.bar(dow, x="DayOfWeek", y="Sales",
            title="Vendas Médias por Dia da Semana",
            labels={"DayOfWeek":"","Sales":"Vendas (€)"},
            color="Sales", color_continuous_scale="Blues", template="plotly_white")
        fig_dow.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_b:
        m_map = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                 7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
        mon = store_full.groupby("Month")["Sales"].mean().reset_index()
        mon["Month"] = mon["Month"].map(m_map)
        fig_mon = px.bar(mon, x="Month", y="Sales",
            title="Vendas Médias por Mês",
            labels={"Month":"","Sales":"Vendas (€)"},
            color="Sales", color_continuous_scale="Greens", template="plotly_white")
        fig_mon.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_mon, use_container_width=True)

    # Impacto da promoção
    promo_avg = store_full.groupby("Promo")["Sales"].mean().reset_index()
    promo_avg["Promo"] = promo_avg["Promo"].map({0: "Sem Promoção", 1: "Com Promoção"})
    fig_promo = px.bar(promo_avg, x="Promo", y="Sales",
        title="Impacto da Promoção nas Vendas Médias",
        labels={"Promo":"","Sales":"Vendas Médias (€)"},
        color="Promo", color_discrete_map={"Sem Promoção":"#d62728","Com Promoção":"#2ca02c"},
        template="plotly_white")
    st.plotly_chart(fig_promo, use_container_width=True)

    # Distribuição de vendas
    fig_hist = px.histogram(series, x="y", nbins=60,
        title="Distribuição das Vendas Diárias",
        labels={"y":"Vendas (€)"},
        color_discrete_sequence=["#1f77b4"], template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Tendência mensal
    monthly = series.copy()
    monthly["mes"] = monthly["ds"].dt.to_period("M")
    monthly_avg = monthly.groupby("mes")["y"].mean().reset_index()
    monthly_avg["mes"] = monthly_avg["mes"].dt.to_timestamp()
    fig_trend = px.line(monthly_avg, x="mes", y="y",
        title="Tendência — Média Mensal de Vendas",
        labels={"mes":"Data","y":"Vendas Médias (€)"}, template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True)


# ── TAB 3: COMPONENTES ─────────────────────────────────────────
with tab3:
    st.subheader("Decomposição em Componentes")
    st.markdown(
        "O Prophet decompõe a série em **tendência**, **sazonalidade semanal**, "
        "**sazonalidade anual** e **efeito da Promoção**."
    )
    future_full = model.make_future_dataframe(periods=forecast_days, freq="D")
    # Preenche Promo para o plot de componentes (usa padrão histórico)
    promo_map = series.set_index("ds")["Promo"].to_dict()
    future_full["Promo"] = future_full["ds"].map(promo_map)
    store_copy = series.copy()
    store_copy["weekday"] = store_copy["ds"].dt.weekday
    rate = store_copy.groupby("weekday")["Promo"].mean()
    mask = future_full["Promo"].isna()
    future_full.loc[mask, "Promo"] = future_full.loc[mask, "ds"].dt.weekday.map(rate).fillna(0).round()
    future_full["Promo"] = future_full["Promo"].fillna(0).astype(int)

    forecast_full = model.predict(future_full)
    fig_comp = model.plot_components(forecast_full)
    st.pyplot(fig_comp, use_container_width=True)
    st.info(
        "💡 O componente **'Promo'** mostra o uplift médio de vendas nos dias com promoção. "
        "Valores positivos = aquele fator impulsiona vendas."
    )


# ── TAB 4: AVALIAÇÃO ───────────────────────────────────────────
with tab4:
    st.subheader("Avaliação do Modelo")
    st.code(metrics_report(metrics, store_id))

    st.markdown("#### Comparativo de modelos (experimento documentado)")
    comparison = pd.DataFrame({
        "Configuração": [
            "Baseline (multiplicative, sem Promo)",
            "Additive, sem Promo",
            "Additive + Promo ✅ (atual)",
        ],
        "MAPE (%)": [15.54, 15.47, 7.54],
        "MAE (€)":  [668,   667,   331],
        "RMSE (€)": [755,   756,   415],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    fig_bar = px.bar(comparison, x="Configuração", y="MAPE (%)",
        title="MAPE por Configuração — Experimento de Seleção de Modelo",
        color="MAPE (%)", color_continuous_scale="RdYlGn_r",
        template="plotly_white", text="MAPE (%)")
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    if len(test_merged):
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=test_merged["ds"], y=test_merged["y"],
            mode="lines+markers", name="Real", line=dict(color="#ff7f0e", width=2)))
        fig_eval.add_trace(go.Scatter(x=test_merged["ds"], y=test_merged["yhat"],
            mode="lines+markers", name="Previsto (Promo)", line=dict(color="#2ca02c", width=2, dash="dash")))
        fig_eval.update_layout(
            title=f"Real vs Previsto — {test_days} dias de teste",
            xaxis_title="Data", yaxis_title="Vendas (€)",
            hovermode="x unified", template="plotly_white", height=380)
        st.plotly_chart(fig_eval, use_container_width=True)

        test_merged["Resíduo"] = test_merged["y"] - test_merged["yhat"]
        fig_res = px.bar(test_merged, x="ds", y="Resíduo",
            title="Resíduos (Real − Previsto)",
            labels={"ds":"Data","Resíduo":"€"},
            color="Resíduo", color_continuous_scale=["#d62728","#aec7e8","#2ca02c"],
            template="plotly_white")
        fig_res.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_res, use_container_width=True)

st.markdown("---")
st.markdown(
    "<center>TCC — Ciência de Dados | Prophet + Regressor Promo | Rossmann Store Sales</center>",
    unsafe_allow_html=True
)
