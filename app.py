from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Efficient Frontier | AAPL - TSLA",
    page_icon="📈",
    layout="wide",
)

TICKERS = ["AAPL", "TSLA"]
ASSET_NAMES = {"AAPL": "Apple", "TSLA": "Tesla"}
TRADING_DAYS = 252


st.markdown(
    """
    <style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    [data-testid="stMetric"] {
        background: #131a24;
        border: 1px solid #283345;
        padding: 18px;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"] { color: #8b9bb4; }
    h1, h2, h3 { color: #f0f4fa; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(years=10):
    """Download daily adjusted closing prices from Yahoo Finance."""
    end_date = date.today() + timedelta(days=1)
    start_date = date.today() - timedelta(days=365 * years + 5)

    raw = yf.download(
        tickers=TICKERS,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            "Yahoo Finance no devolvió datos. Intenta nuevamente más tarde."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("La descarga no contiene precios de cierre.")
        close = raw.xs("Close", axis=1, level=0)
    else:
        if "Close" not in raw.columns:
            raise ValueError("La descarga no contiene precios de cierre.")
        close = raw["Close"]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=TICKERS[0])

    close = close.reindex(columns=TICKERS)
    close = close.apply(pd.to_numeric, errors="coerce")
    close = close.dropna(how="any").sort_index()

    if close.empty or len(close) < 252:
        raise ValueError(
            "No hay suficientes observaciones comunes para calcular "
            "la frontera eficiente."
        )

    return close


def calculate_portfolios(prices, n_portfolios=100):
    """Calculate long-only portfolios using daily returns and annualized stats."""
    daily_returns = prices.pct_change().dropna()
    mean_daily = daily_returns.mean()
    cov_daily = daily_returns.cov()

    mean_annual = mean_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS

    weights_aapl = np.linspace(0.0, 1.0, n_portfolios)
    records = []

    for weight_aapl in weights_aapl:
        weight_tsla = 1.0 - weight_aapl
        weights = np.array([weight_aapl, weight_tsla])

        expected_return = float(weights @ mean_annual.to_numpy())
        variance = float(weights @ cov_annual.to_numpy() @ weights)
        volatility = np.sqrt(max(variance, 0.0))

        records.append(
            {
                "AAPL (%)": weight_aapl * 100,
                "TSLA (%)": weight_tsla * 100,
                "Retorno anual (%)": expected_return * 100,
                "Desvio anual (%)": volatility * 100,
            }
        )

    return daily_returns, mean_annual, cov_annual, pd.DataFrame(records)


def add_risk_metrics(portfolios, risk_free_rate):
    result = portfolios.copy()
    excess_return = result["Retorno anual (%)"] / 100 - risk_free_rate
    volatility = result["Desvio anual (%)"] / 100
    result["Sharpe"] = np.where(
        volatility > 0, excess_return / volatility, np.nan
    )
    return result


st.title("EFFICIENT FRONTIER")
st.caption("MARKOWITZ PORTFOLIO ANALYTICS  |  AAPL · TSLA  |  DAILY DATA")

with st.sidebar:
    st.header("Configuración")
    years = st.selectbox(
        "Historia de precios",
        options=[10],
        index=0,
        format_func=lambda x: f"Últimos {x} años",
    )
    n_portfolios = 100
    st.metric("Portafolios", n_portfolios)
    risk_free_rate_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=0.25,
        help="Se usa para el ratio de Sharpe, no para el retorno esperado.",
    )
    st.caption("Datos diarios de Yahoo Finance.")
    st.caption("Anualización: 252 ruedas bursátiles.")

try:
    with st.spinner("Descargando precios históricos de Yahoo Finance..."):
        prices = download_prices(years)
    daily_returns, mean_annual, cov_annual, portfolios = calculate_portfolios(
        prices, n_portfolios
    )
    portfolios = add_risk_metrics(portfolios, risk_free_rate_pct / 100)
except Exception as exc:
    st.error(f"No se pudieron calcular los datos: {exc}")
    st.info(
        "Verifica tu conexión a internet y que las dependencias estén instaladas."
    )
    st.stop()

idx_min_var = portfolios["Desvio anual (%)"].idxmin()
idx_max_sharpe = portfolios["Sharpe"].idxmax()
min_var = portfolios.loc[idx_min_var]
max_sharpe = portfolios.loc[idx_max_sharpe]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observaciones diarias", f"{len(daily_returns):,}")
col2.metric("Retorno anual AAPL", f"{mean_annual['AAPL'] * 100:.2f}%")
col3.metric("Retorno anual TSLA", f"{mean_annual['TSLA'] * 100:.2f}%")
col4.metric(
    "Correlación AAPL / TSLA",
    f"{daily_returns['AAPL'].corr(daily_returns['TSLA']):.3f}",
)

st.divider()
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=portfolios["Desvio anual (%)"],
        y=portfolios["Retorno anual (%)"],
        mode="lines+markers",
        name="Portafolios",
        customdata=portfolios[
            ["AAPL (%)", "TSLA (%)", "Sharpe"]
        ].to_numpy(),
        line=dict(width=2.5, color="#39a0ff"),
        marker=dict(
            size=6,
            color=portfolios["Retorno anual (%)"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Retorno (%)"),
        ),
        hovertemplate=(
            "<b>Portafolio</b><br>"
            "Desvío anual: %{x:.2f}%<br>"
            "Retorno anual: %{y:.2f}%<br>"
            "AAPL: %{customdata[0]:.2f}%<br>"
            "TSLA: %{customdata[1]:.2f}%<br>"
            "Sharpe: %{customdata[2]:.3f}<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[min_var["Desvio anual (%)"]],
        y=[min_var["Retorno anual (%)"]],
        mode="markers",
        name="Mínima varianza",
        marker=dict(
            size=14,
            color="#ffbd4a",
            symbol="diamond",
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "Mínima varianza<br>Desvío: %{x:.2f}%<br>"
            "Retorno: %{y:.2f}%<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[max_sharpe["Desvio anual (%)"]],
        y=[max_sharpe["Retorno anual (%)"]],
        mode="markers",
        name="Máximo Sharpe",
        marker=dict(
            size=14,
            color="#ff5d73",
            symbol="star",
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "Máximo Sharpe<br>Desvío: %{x:.2f}%<br>"
            "Retorno: %{y:.2f}%<extra></extra>"
        ),
    )
)

for ticker in TICKERS:
    target_weight = 100.0 if ticker == "AAPL" else 0.0
    row = portfolios.iloc[
        (portfolios["AAPL (%)"] - target_weight).abs().argmin()
    ]
    fig.add_trace(
        go.Scatter(
            x=[row["Desvio anual (%)"]],
            y=[row["Retorno anual (%)"]],
            mode="markers+text",
            name=ASSET_NAMES[ticker],
            text=[ticker],
            textposition="top center",
            marker=dict(size=10, symbol="circle-open"),
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=620,
    title="Frontera de Markowitz — Retorno vs. Riesgo anual",
    xaxis_title="Desvío estándar anual (%)",
    yaxis_title="Retorno esperado anual (%)",
    hovermode="closest",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
    ),
    margin=dict(l=30, r=30, t=90, b=40),
)
fig.update_xaxes(showgrid=True, gridcolor="#273244")
fig.update_yaxes(showgrid=True, gridcolor="#273244")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Portafolios destacados")
a, b = st.columns(2)

with a:
    st.markdown("#### Mínima varianza")
    st.metric("Retorno anual", f"{min_var['Retorno anual (%)']:.2f}%")
    st.metric("Desvío anual", f"{min_var['Desvio anual (%)']:.2f}%")
    st.write(
        f"AAPL: {min_var['AAPL (%)']:.2f}%  |  "
        f"TSLA: {min_var['TSLA (%)']:.2f}%"
    )

with b:
    st.markdown("#### Máximo Sharpe")
    st.metric("Retorno anual", f"{max_sharpe['Retorno anual (%)']:.2f}%")
    st.metric("Desvío anual", f"{max_sharpe['Desvio anual (%)']:.2f}%")
    st.metric("Ratio de Sharpe", f"{max_sharpe['Sharpe']:.3f}")
    st.write(
        f"AAPL: {max_sharpe['AAPL (%)']:.2f}%  |  "
        f"TSLA: {max_sharpe['TSLA (%)']:.2f}%"
    )

st.divider()
st.subheader("Matriz de los 100 portafolios")
display_df = portfolios.copy()
for column in [
    "AAPL (%)",
    "TSLA (%)",
    "Retorno anual (%)",
    "Desvio anual (%)",
    "Sharpe",
]:
    display_df[column] = display_df[column].round(4)

st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

csv_data = portfolios.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Descargar los 100 portafolios en CSV",
    data=csv_data,
    file_name="frontera_eficiente_AAPL_TSLA.csv",
    mime="text/csv",
)

with st.expander("Consultar precios y retornos diarios"):
    tab1, tab2 = st.tabs(["Precios ajustados", "Retornos diarios"])
    with tab1:
        st.dataframe(prices, use_container_width=True)
    with tab2:
        st.dataframe(daily_returns.tail(100), use_container_width=True)

st.caption(
    f"Período descargado: {prices.index.min().date()} a "
    f"{prices.index.max().date()}. Los resultados son estimaciones "
    "históricas, no una garantía de rendimientos futuros."
)
