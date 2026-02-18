import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import math
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from supabase import create_client

# =========================
# SUPABASE CONFIG
# =========================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_strategy(name, spot_entry, legs):
    data = {
        "name": name,
        "spot_entry": spot_entry,
        "legs": legs
    }
    supabase.table("strategies").insert(data).execute()

def delete_strategy(strategy_id):
    supabase.table("strategies") \
        .delete() \
        .eq("id", strategy_id) \
        .execute()


# =========================
# CÁLCULOS DE CARTEIRA
# =========================

def calculate_entry_value(legs):
    total = 0

    for leg in legs:
        premium = leg["premium_entry_usd"] * leg["quantity"]

        if leg["side"] == "buy":
            total -= premium
        else:
            total += premium

    return total


def calculate_strategy_values(legs, spot_now):

    total_pl = 0
    current_value = 0

    for leg in legs:

        strike = leg["strike"]
        qty = leg["quantity"]
        entry = leg["premium_entry_usd"]

        if leg["type"] == "call":
            intrinsic = max(spot_now - strike, 0)
        else:
            intrinsic = max(strike - spot_now, 0)

        # valor atual da perna
        value_now = intrinsic * qty
        current_value += value_now

        # P/L
        if leg["side"] == "buy":
            pl = (intrinsic - entry) * qty
        else:
            pl = (entry - intrinsic) * qty

        total_pl += pl

    return current_value, total_pl


def average_iv(legs):
    ivs = [leg.get("iv_entry", 0) for leg in legs]
    if not ivs:
        return 0
    return sum(ivs) / len(ivs)


def load_strategies():
    response = supabase.table("strategies") \
        .select("*") \
        .order("created_at", desc=True) \
        .execute()

    return response.data


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
            "quantity": quantity,
            "premium": premium,                     # ETH
            "premium_usd": premium * spot_price,    # USD
            "premium_entry_usd": premium * spot_price,
            "iv_entry": iv,
            "enabled": True
    })




# =========================
# COLUNA DIREITA — CARTEIRA
# =========================
with right:

    st.subheader("Pernas da estratégia")

    legs_to_remove = None

    for i, leg in enumerate(st.session_state.legs):

        # Layout das colunas
        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 1])

        # -------- Status --------
        leg["enabled"] = col1.checkbox(
            "On",
            value=leg.get("enabled", True),
            key=f"enabled_{i}"
        )

        # -------- Informações básicas --------
        side = leg.get("side", "").upper()
        opt_type = leg.get("type", "").upper()
        strike = leg.get("strike", 0)
        qty = leg.get("quantity", 0)

        col2.write(f"{side} {opt_type}")
        col3.write(f"Strike {strike}")
        col4.write(f"Qty {qty}")

        # -------- Premium (compatível com dados antigos) --------
        premium_eth = leg.get("premium") or 0
        premium_usd = (
            leg.get("premium_usd")
            or leg.get("premium_entry_usd")
            or 0
        )

        col5.write(
            f"{premium_eth:.4f} ETH (~${premium_usd:.2f})"
        )

        # -------- Remover perna --------
        if col6.button("❌", key=f"remove_{i}"):
            legs_to_remove = i

    # Remove após o loop
    if legs_to_remove is not None:
        st.session_state.legs.pop(legs_to_remove)

    # =========================
    # CUSTO TOTAL (baseado no valor de entrada)
    # =========================
    total_cost = 0

    for leg in st.session_state.legs:

        if not leg.get("enabled", True):
            continue

        premium_usd = (
            leg.get("premium_usd")
            or leg.get("premium_entry_usd")
            or 0
        )

        qty = leg.get("quantity", 0)

        cost = premium_usd * qty

        if leg.get("side") == "buy":
            total_cost -= cost   # débito
        else:
            total_cost += cost   # crédito

    st.metric("Custo total (USD)", f"${total_cost:,.2f}")

  

# =========================
# SALVAR ESTRATÉGIA
# =========================
st.markdown("---")
st.subheader("Salvar estratégia")

strategy_name = st.text_input(
    "Nome da estratégia",
    placeholder="Ex: Short Put ETH 1800"
)

if st.button("Salvar estratégia"):

    active_legs = [leg for leg in st.session_state.legs if leg["enabled"]]

    if not active_legs:
        st.warning("Nenhuma perna ativa para salvar")
    elif strategy_name.strip() == "":
        st.warning("Digite um nome para a estratégia")
    else:
        try:
            save_strategy(
                name=strategy_name,
                spot_entry=spot_price,
                legs=active_legs
            )
            st.success("Estratégia salva com sucesso!")
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")



# =========================
# SIMULAÇÃO
# =========================

if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

if st.button("Simular estratégia"):
    st.session_state.run_simulation = True


if st.session_state.run_simulation:

    # Pernas ativas (seguro para dados antigos)
    active_legs = [leg for leg in st.session_state.legs if leg.get("enabled", True)]

    if not active_legs:
        st.warning("Nenhuma perna ativa")
        st.stop()

    # IV média
    iv_list = [leg.get("iv_entry", 0) for leg in active_legs]
    iv = sum(iv_list) / len(iv_list) if iv_list else 0.8

    # =========================
    # CÁLCULO DO PAYOFF
    # =========================
    spot = float(spot_price)
    prices = np.linspace(0.01, spot * 2, 400)
    total_payoff = np.zeros_like(prices, dtype=float)

    for leg in active_legs:

        # Normalização
        leg_type = leg.get("type")
        side = leg.get("side")
        strike = float(leg.get("strike", 0))
        qty = float(leg.get("quantity", 0))

        premium_usd = (
            leg.get("premium_usd")
            or leg.get("premium_entry_usd")
            or 0
        )
        premium_usd = float(premium_usd)

        # Intrínseco
        if leg_type == "call":
            intrinsic = np.maximum(prices - strike, 0)
        else:
            intrinsic = np.maximum(strike - prices, 0)

        # Payoff
        if side == "buy":
            payoff_leg = intrinsic - premium_usd
        else:
            payoff_leg = premium_usd - intrinsic

        total_payoff += payoff_leg * qty

    payoff = total_payoff

    # =========================
    # MÉTRICAS
    # =========================
    max_profit = float(np.max(payoff))
    max_loss = float(np.min(payoff))

    current_index = np.abs(prices - spot).argmin()
    current_payoff = payoff[current_index]

    breakeven_points = []
    for i in range(1, len(payoff)):
        if payoff[i-1] * payoff[i] < 0:
            breakeven_points.append(prices[i])

    # =========================
    # PROBABILIDADE
    # =========================
    now = time.time()
    time_to_expiry = max((expiration / 1000 - now) / (365 * 24 * 3600), 0.0001)
    std = spot * iv * np.sqrt(time_to_expiry)

    def norm_cdf(x, mu, sigma):
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

    prob_profit = 0
    for i in range(len(prices) - 1):
        if payoff[i] > 0 or payoff[i + 1] > 0:
            p1 = norm_cdf(prices[i], spot, std)
            p2 = norm_cdf(prices[i + 1], spot, std)
            prob_profit += max(p2 - p1, 0)

    prob_profit *= 100

    # =========================
    # PAINEL
    # =========================
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P/L Atual", f"${current_payoff:,.2f}")
    c2.metric("Lucro Máx", f"${max_profit:,.2f}")
    c3.metric("Perda Máx", f"${max_loss:,.2f}")
    c4.metric(
        "Break-even",
        ", ".join([f"{be:.0f}" for be in breakeven_points]) if breakeven_points else "-"
    )
    c5.metric("Prob. Lucro", f"{prob_profit:.1f}%")

        # =========================
    # GRÁFICO DE PAYOFF
    # =========================
    fig = go.Figure()

    profit = np.maximum(payoff, 0)
    loss = np.minimum(payoff, 0)

    fig.add_trace(go.Scatter(
        x=prices,
        y=profit,
        fill='tozeroy',
        mode='lines',
        name='Lucro',
        line=dict(color='green'),
        hovertemplate="Preço: $%{x:,.0f}<br>Lucro: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=prices,
        y=loss,
        fill='tozeroy',
        mode='lines',
        name='Prejuízo',
        line=dict(color='red'),
        hovertemplate="Preço: $%{x:,.0f}<br>Prejuízo: $%{y:,.2f}<extra></extra>"
    ))

    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")

    for be in breakeven_points:
        fig.add_vline(x=be, line_dash="dot")

    fig.update_layout(
        template="plotly_dark",
        height=500,
        hoverlabel=dict(font_size=16)
    )

    fig.update_xaxes(range=[spot * 0.5, spot * 1.5])

    st.plotly_chart(fig, use_container_width=True)



# =========================
# CARTEIRA
# =========================
st.markdown("---")
st.subheader("📊 Carteira de Estratégias")

try:
    strategies = load_strategies()

    if not strategies:
        st.info("Nenhuma estratégia salva.")
    else:
        for strat in strategies:

            legs = strat["legs"]

            # =========================
            # CÁLCULOS
            # =========================
            entry_value = 0
            current_value = 0
            total_pl = 0
            iv_list = []
            strikes = []

            for leg in legs:

                strike = leg["strike"]
                qty = leg["quantity"]
                strikes.append(strike)

                # Premium de entrada (USD)
                premium_entry = leg.get("premium_entry_usd", 0)

                # IV na entrada
                iv_list.append(leg.get("iv_entry", 0))

                # Crédito / Débito na entrada
                if leg["side"] == "buy":
                    entry_value -= premium_entry * qty
                else:
                    entry_value += premium_entry * qty

                # Valor atual (intrínseco)
                if leg["type"] == "call":
                    intrinsic = max(spot_price - strike, 0)
                else:
                    intrinsic = max(strike - spot_price, 0)

                current_value += intrinsic * qty

                # P/L
                if leg["side"] == "buy":
                    pl_leg = (intrinsic - premium_entry) * qty
                else:
                    pl_leg = (premium_entry - intrinsic) * qty

                total_pl += pl_leg

            avg_iv = sum(iv_list) / len(iv_list) if iv_list else 0

            # Strikes únicos (ordenados)
            strikes_text = ", ".join([str(int(s)) for s in sorted(set(strikes))])

            # =========================
            # EXIBIÇÃO
            # =========================
            with st.expander(f"{strat['name']} | Strikes: {strikes_text}"):

                col1, col2, col3, col4 = st.columns([1,1,1,0.7])
              
                # Entrada
                if entry_value >= 0:
                    col1.success(f"Crédito entrada: +${entry_value:,.2f}")
                else:
                    col1.error(f"Débito entrada: ${entry_value:,.2f}")

                # Atual
                col2.metric("Valor atual", f"${current_value:,.2f}")

                # P/L
                if total_pl >= 0:
                    col3.success(f"P/L: +${total_pl:,.2f}")
                else:
                    col3.error(f"P/L: ${total_pl:,.2f}")

                st.caption(f"IV média na entrada: {avg_iv*100:.1f}%")
                st.caption(f"Criada em: {strat['created_at']}")

                st.markdown("**Pernas:**")
                for leg in legs:
                    st.write(
                        f"{leg['side'].upper()} {leg['type'].upper()} | "
                        f"Strike {leg['strike']} | "
                        f"Qty {leg['quantity']}"
                    )
                  # Botão excluir
                if col4.button("🗑", key=f"delete_{strat['id']}"):
                    try:
                        delete_strategy(strat["id"])
                        st.success("Estratégia excluída!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao excluir: {e}")

except Exception as e:
    st.error(f"Erro ao carregar carteira: {e}")


    
     

