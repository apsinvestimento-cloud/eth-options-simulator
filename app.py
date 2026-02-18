import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import math
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="ETH Options Simulator", layout="wide")

st.title("ETH Options Simulator")

# Atualiza a cada 30 segundos
st_autorefresh(interval=30 * 1000, key="refresh")

# =========================
# SESSION STATE INIT
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = []

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False


# =========================
# CACHE (importante para Cloud)
# =========================
@st.cache_data(ttl=30)
def get_spot():
    data = requests.get(
        "https://www.deribit.com/api/v2/public/ticker?instrument_name=ETH-PERPETUAL"
    ).json()
    return data["result"]["last_price"]


@st.cache_data(ttl=60)
def get_instruments():
    return requests.get(
        "https://www.deribit.com/api/v2/public/get_instruments?currency=ETH&kind=option"
    ).json()["result"]


@st.cache_data(ttl=30)
def get_option_market(instrument_name):
    data = requests.get(
        f"https://www.deribit.com/api/v2/public/ticker?instrument_name={instrument_name}"
    ).json()["result"]

    premium = data.get("mark_price", 0)
    iv = data.get("mark_iv", 80) / 100

    return premium, iv


# =========================
# SPOT
# =========================# =========================
# SPOT
# =========================
spot_price = get_spot()
st.metric("ETH Spot", f"${spot_price:,.2f}")

st.write("Simulador de estratégias multi-perna com dados da Deribit")

# =========================
# LAYOUT PRINCIPAL
# =========================
left, right = st.columns([1, 1])

# =========================
# COLUNA ESQUERDA — MONTAR PERNA
# =========================
with left:

    st.subheader("Adicionar perna")

    instruments = get_instruments()

    expirations = sorted(list(set([inst["expiration_timestamp"] for inst in instruments])))

    exp_list = [
        {
            "timestamp": exp,
            "date": datetime.utcfromtimestamp(exp / 1000).strftime("%Y-%m-%d")
        }
        for exp in expirations
    ]

    exp_dates = [exp["date"] for exp in exp_list]
    selected_date = st.selectbox("Vencimento", exp_dates)

    expiration = next(
        exp["timestamp"] for exp in exp_list if exp["date"] == selected_date
    )

    option_type = st.selectbox("Tipo", ["call", "put"])

    filtered = [
        inst for inst in instruments
        if inst["expiration_timestamp"] == expiration
        and inst["option_type"] == option_type
    ]

    strike_list = sorted([inst["strike"] for inst in filtered])

    atm_strike = min(strike_list, key=lambda x: abs(x - spot_price))
    atm_index = strike_list.index(atm_strike)

    strike = st.selectbox(
        "Strike (ATM sugerido)",
        strike_list,
        index=atm_index
    )

    instrument = next(inst for inst in filtered if inst["strike"] == strike)

    premium, iv = get_option_market(instrument["instrument_name"])
    premium_usd = premium * spot_price

    st.write(f"IV: {iv*100:.1f}%")
    st.write(f"Premium: {premium:.4f} ETH")

    side = st.selectbox("Posição", ["buy", "sell"])

    quantity = st.number_input(
        "Quantidade (contratos)",
        min_value=0.1,
        value=0.1,
        step=0.1,
        format="%.1f"
    )

    position_cost = premium * quantity * spot_price

    if side == "buy":
        st.error(f"Débito: -${position_cost:,.2f}")
    else:
        st.success(f"Crédito: +${position_cost:,.2f}")

    if st.button("Adicionar perna"):
        st.session_state.legs.append({
            "type": option_type,
            "side": side,
            "strike": strike,
            "premium": premium,
            "premium_usd": premium_usd,
            "quantity": quantity,
            "enabled": True
        })


# =========================
# COLUNA DIREITA — CARTEIRA
# =========================
with right:

    st.subheader("Pernas da estratégia")

    legs_to_remove = None

    for i, leg in enumerate(st.session_state.legs):

        col1, col2, col3, col4, col5, col6 = st.columns([1,2,2,2,2,1])

        leg["enabled"] = col1.checkbox(
            "On",
            value=leg.get("enabled", True),
            key=f"enabled_{i}"
        )

        col2.write(f"{leg['side'].upper()} {leg['type'].upper()}")
        col3.write(f"Strike {leg['strike']}")
        col4.write(f"Qty {leg['quantity']}")
        col5.write(
            f"{leg['premium']:.4f} ETH (~${leg.get('premium_usd', leg['premium']*spot_price):.2f})"
        )

        if col6.button("❌", key=f"remove_{i}"):
            legs_to_remove = i

    if legs_to_remove is not None:
        st.session_state.legs.pop(legs_to_remove)

    # Custo total
    total_cost = 0
    for leg in st.session_state.legs:
        if not leg["enabled"]:
            continue

        cost = leg["premium"] * leg["quantity"] * spot_price

        if leg["side"] == "buy":
            total_cost += cost
        else:
            total_cost -= cost

    st.metric("Custo total (USD)", f"${total_cost:,.2f}")

    simulate_button = st.button("Simular estratégia")



# =========================
# SIMULAÇÃO
# =========================

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

if simulate_button:
    st.session_state.run_simulation = True


if st.session_state.run_simulation:

    # Pernas ativas
    active_legs = [leg for leg in st.session_state.legs if leg["enabled"]]

    if not active_legs:
        st.warning("Nenhuma perna ativa")
        st.stop()

    # =========================
    # CÁLCULO DO PAYOFF
    # =========================
    spot = spot_price
    prices = np.linspace(0.01, spot * 2, 400)
    total_payoff = np.zeros_like(prices)

    for leg in active_legs:

        # Intrínseco
        if leg["type"] == "call":
            intrinsic = np.maximum(prices - leg["strike"], 0)
        else:
            intrinsic = np.maximum(leg["strike"] - prices, 0)

        # Premium em USD
        premium_usd = leg["premium"] * spot

        # Payoff
        if leg["side"] == "buy":
            payoff_leg = intrinsic - premium_usd
        else:
            payoff_leg = premium_usd - intrinsic

        payoff_leg = payoff_leg * leg["quantity"]

        total_payoff += payoff_leg

    payoff = total_payoff


    # (segue métricas e gráfico normalmente)


    # =========================
    # MÉTRICAS
    # =========================
    max_profit = max(payoff)
    max_loss = min(payoff)

    current_index = min(range(len(prices)), key=lambda i: abs(prices[i] - spot))
    current_payoff = payoff[current_index]

    breakeven_points = []
    for i in range(1, len(payoff)):
        if payoff[i-1] * payoff[i] < 0:
            breakeven_points.append(prices[i])

    # Probabilidade
    now = time.time()
    time_to_expiry = max((expiration / 1000 - now) / (365 * 24 * 3600), 0.0001)
    std = spot * iv * np.sqrt(time_to_expiry)

    def norm_cdf(x, mu, sigma):
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

    prob_profit = 0
    for i in range(len(prices)-1):
        if payoff[i] > 0 or payoff[i+1] > 0:
            p1 = norm_cdf(prices[i], spot, std)
            p2 = norm_cdf(prices[i+1], spot, std)
            prob_profit += max(p2 - p1, 0)

    prob_profit *= 100

    # =========================
    # PAINEL
    # =========================
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P/L Atual", f"${current_payoff:,.2f}")
    c2.metric("Lucro Máx", f"${max_profit:,.2f}")
    c3.metric("Perda Máx", f"${max_loss:,.2f}")
    c4.metric("Break-even", ", ".join([f"{be:.0f}" for be in breakeven_points]) if breakeven_points else "-")
    c5.metric("Prob. Lucro", f"{prob_profit:.1f}%")

       # =========================
    # GRÁFICO
    # =========================
    fig = go.Figure()

    profit = [max(p, 0) for p in payoff]
    loss = [min(p, 0) for p in payoff]

    fig.add_trace(go.Scatter(
        x=prices,
        y=profit,
        fill='tozeroy',
        mode='lines',
        name='Lucro',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=prices,
        y=loss,
        fill='tozeroy',
        mode='lines',
        name='Prejuízo',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=prices,
        y=profit,
        fill='tozeroy',
        mode='lines',
        name='Lucro',
        line=dict(color='green'),
        hovertemplate=
            "Preço: $%{x:,.0f}<br>"
            "Lucro: $%{y:,.2f}"
            "<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=prices,
        y=loss,
        fill='tozeroy',
        mode='lines',
        name='Prejuízo',
        line=dict(color='red'),
        hovertemplate=
            "Preço: $%{x:,.0f}<br>"
            "Prejuízo: $%{y:,.2f}"
            "<extra></extra>"
    ))




    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")

    for be in breakeven_points:
        fig.add_vline(x=be, line_dash="dot")

    # Layout + hover maior
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hoverlabel=dict(
            font_size=16,
            font_family="Arial"
        )
    )

    # Limita o zoom visual do eixo X
    fig.update_xaxes(range=[spot * 0.5, spot * 1.5])

    st.plotly_chart(fig, use_container_width=True)


