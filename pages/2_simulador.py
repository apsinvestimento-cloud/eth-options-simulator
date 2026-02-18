import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import math
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from supabase import create_client
from utils.auth import require_login

# =========================
# PROTEÇÃO
# =========================
require_login()

st.set_page_config(page_title="Simulador", layout="wide")
st.title("📈 Simulador de Opções ETH")

# Atualiza preços automaticamente
st_autorefresh(interval=30 * 1000, key="refresh")

# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

user_id = st.session_state.user.id

# =========================
# SESSION STATE
# =========================
if "legs" not in st.session_state:
    st.session_state.legs = []

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

# =========================
# API DERIBIT
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


spot_price = get_spot()
st.metric("ETH Spot", f"${spot_price:,.2f}")

left, right = st.columns([1,1])

# =========================
# ADICIONAR PERNA
# =========================
with left:

    st.subheader("Adicionar perna")

    instruments = get_instruments()

    expirations = sorted(list(set([i["expiration_timestamp"] for i in instruments])))

    exp_list = [
        {
            "timestamp": e,
            "date": datetime.utcfromtimestamp(e/1000).strftime("%Y-%m-%d")
        }
        for e in expirations
    ]

    exp_dates = [e["date"] for e in exp_list]
    selected_date = st.selectbox("Vencimento", exp_dates)

    expiration = next(e["timestamp"] for e in exp_list if e["date"] == selected_date)

    option_type = st.selectbox("Tipo", ["call", "put"])

    filtered = [
        inst for inst in instruments
        if inst["expiration_timestamp"] == expiration
        and inst["option_type"] == option_type
    ]

    strike_list = sorted([inst["strike"] for inst in filtered])

    strike = st.selectbox("Strike", strike_list)

    instrument = next(inst for inst in filtered if inst["strike"] == strike)

    premium, iv = get_option_market(instrument["instrument_name"])
    premium_usd = premium * spot_price

    st.write(f"IV: {iv*100:.1f}%")
    st.write(f"Premium: {premium:.4f} ETH (~${premium_usd:.2f})")

    side = st.selectbox("Posição", ["buy", "sell"])

    quantity = st.number_input("Quantidade", 0.1, step=0.1)

    if st.button("Adicionar perna"):
        new_leg = {
            "type": option_type,
            "side": side,
            "strike": strike,
            "quantity": float(quantity),
            "premium_usd": premium_usd,
            "premium_entry_usd": premium_usd,
            "iv_entry": iv,
            "instrument_name": instrument["instrument_name"],
            "expiration_timestamp": expiration,
            "expiration_date": selected_date,
            "enabled": True
        }
        st.session_state.legs.append(new_leg)
        st.success("Perna adicionada")


# =========================
# PERNAS DA ESTRATÉGIA
# =========================
with right:

    st.subheader("Pernas")

    remove_index = None

    for i, leg in enumerate(st.session_state.legs):

        c1, c2, c3, c4, c5 = st.columns([1,2,2,2,1])

        c1.write(leg["side"].upper())
        c2.write(f"{leg['type'].upper()} {leg['strike']}")
        c3.write(f"Qty {leg['quantity']}")
        c4.write(f"${leg['premium_entry_usd']:.2f}")

        if c5.button("❌", key=f"del_{i}"):
            remove_index = i

    if remove_index is not None:
        st.session_state.legs.pop(remove_index)
        st.rerun()


# =========================
# BOTÃO SIMULAR
# =========================
st.markdown("---")

if st.button("Simular estratégia"):
    st.session_state.run_simulation = True


# =========================
# SIMULAÇÃO
# =========================
if st.session_state.run_simulation:

    active_legs = [leg for leg in st.session_state.legs if leg.get("enabled", True)]

    if not active_legs:
        st.warning("Nenhuma perna ativa")
        st.stop()

    # IV média
    iv_list = [leg.get("iv_entry", 0) for leg in active_legs]
    iv = sum(iv_list) / len(iv_list) if iv_list else 0.8

    spot = float(spot_price)
    prices = np.linspace(0.01, spot * 2, 400)
    total_payoff = np.zeros_like(prices)

    for leg in active_legs:

        strike = float(leg["strike"])
        qty = float(leg["quantity"])
        premium = float(leg["premium_entry_usd"])
        side = leg["side"]
        leg_type = leg["type"]

        if leg_type == "call":
            intrinsic = np.maximum(prices - strike, 0)
        else:
            intrinsic = np.maximum(strike - prices, 0)

        if side == "buy":
            payoff_leg = intrinsic - premium
        else:
            payoff_leg = premium - intrinsic

        total_payoff += payoff_leg * qty

    payoff = total_payoff

    # =========================
    # MÉTRICAS
    # =========================
    max_profit = np.max(payoff)
    max_loss = np.min(payoff)

    current_index = np.abs(prices - spot).argmin()
    current_payoff = payoff[current_index]

    breakeven = []
    for i in range(1, len(payoff)):
        if payoff[i-1] * payoff[i] < 0:
            breakeven.append(prices[i])

    # Probabilidade
    now = time.time()
    time_to_expiry = max((expiration/1000 - now)/(365*24*3600), 0.0001)
    std = spot * iv * np.sqrt(time_to_expiry)

    def norm_cdf(x, mu, sigma):
        return 0.5 * (1 + math.erf((x-mu)/(sigma*math.sqrt(2))))

    prob_profit = 0
    for i in range(len(prices)-1):
        if payoff[i] > 0 or payoff[i+1] > 0:
            prob_profit += max(norm_cdf(prices[i+1], spot, std) - norm_cdf(prices[i], spot, std), 0)

    prob_profit *= 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P/L Atual", f"${current_payoff:,.2f}")
    c2.metric("Lucro Máx", f"${max_profit:,.2f}")
    c3.metric("Perda Máx", f"${max_loss:,.2f}")
    c4.metric("Break-even", ", ".join([f"{b:.0f}" for b in breakeven]) if breakeven else "-")
    c5.metric("Prob. Lucro", f"{prob_profit:.1f}%")

    # =========================
    # GRÁFICO
    # =========================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices,
        y=np.maximum(payoff,0),
        fill="tozeroy",
        mode="lines",
        name="Lucro"
    ))

    fig.add_trace(go.Scatter(
        x=prices,
        y=np.minimum(payoff,0),
        fill="tozeroy",
        mode="lines",
        name="Prejuízo"
    ))

    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")

    for b in breakeven:
        fig.add_vline(x=b, line_dash="dot")

    fig.update_layout(template="plotly_dark", height=500)
    fig.update_xaxes(range=[spot*0.5, spot*1.5])

    st.plotly_chart(fig, use_container_width=True)
