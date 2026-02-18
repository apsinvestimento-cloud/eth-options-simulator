import streamlit as st
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from supabase import create_client
from utils.auth import require_login

# =========================
# PROTEÇÃO
# =========================
require_login()

st.set_page_config(page_title="Carteira", layout="wide")
st.title("📊 Carteira")

# Atualiza automaticamente
st_autorefresh(interval=30 * 1000, key="refresh")

# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

user_id = st.session_state.user.id

# =========================
# DADOS DE MERCADO
# =========================
@st.cache_data(ttl=30)
def get_spot():
    data = requests.get(
        "https://www.deribit.com/api/v2/public/ticker?instrument_name=ETH-PERPETUAL"
    ).json()
    return data["result"]["last_price"]

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

# =========================
# CARREGAR ESTRATÉGIAS (do usuário)
# =========================
def load_strategies():
    response = supabase.table("strategies") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return response.data


# =========================
# CARTEIRA
# =========================
try:
    strategies = load_strategies()

    portfolio_entry_total = 0
    portfolio_current_total = 0
    portfolio_pl_total = 0
    portfolio_realized_pl = 0  # futuro

    if not strategies:
        st.info("Nenhuma estratégia salva.")
        st.stop()

    # =========================
    # PROCESSAR ESTRATÉGIAS
    # =========================
    processed_strategies = []

    for strat in strategies:

        legs = strat["legs"]
        entry_value = 0
        current_value = 0
        total_pl = 0
        iv_list = []
        strikes = []

        for leg in legs:

            strike = leg["strike"]
            qty = float(leg["quantity"])
            strikes.append(strike)

            premium_entry = float(leg.get("premium_entry_usd", 0))
            iv_list.append(leg.get("iv_entry", 0))

            # Entrada
            if leg["side"] == "buy":
                entry_value -= premium_entry * qty
            else:
                entry_value += premium_entry * qty

            # Valor atual
            instrument_name = leg.get("instrument_name")

            if instrument_name:
                premium_now, _ = get_option_market(instrument_name)
                value_now = premium_now * spot_price * qty
            else:
                if leg["type"] == "call":
                    intrinsic = max(spot_price - strike, 0)
                else:
                    intrinsic = max(strike - spot_price, 0)
                value_now = intrinsic * qty

            current_value += value_now

            # P/L
            if leg["side"] == "buy":
                pl_leg = value_now - (premium_entry * qty)
            else:
                pl_leg = (premium_entry * qty) - value_now

            total_pl += pl_leg

        # Totais da carteira
        portfolio_entry_total += entry_value
        portfolio_current_total += current_value
        portfolio_pl_total += total_pl

        processed_strategies.append({
            "strat": strat,
            "entry_value": entry_value,
            "current_value": current_value,
            "total_pl": total_pl,
            "avg_iv": sum(iv_list) / len(iv_list) if iv_list else 0,
            "strikes_text": ", ".join([str(int(s)) for s in sorted(set(strikes))])
        })

    # =========================
    # TOPO DA CARTEIRA
    # =========================
    col_title, col_pl, col_real = st.columns([2,1,1])

    col_title.subheader("Carteira de Estratégias")

    if portfolio_pl_total >= 0:
        col_pl.success(f"P/L Atual: +${portfolio_pl_total:,.2f}")
    else:
        col_pl.error(f"P/L Atual: ${portfolio_pl_total:,.2f}")

    col_real.info(f"Realizado: ${portfolio_realized_pl:,.2f}")

    # =========================
    # LISTA DE ESTRATÉGIAS
    # =========================
    for data in processed_strategies:

        strat = data["strat"]
        entry_value = data["entry_value"]
        current_value = data["current_value"]
        total_pl = data["total_pl"]
        avg_iv = data["avg_iv"]
        strikes_text = data["strikes_text"]
        legs = strat["legs"]

        with st.expander(f"{strat['name']} | Strikes: {strikes_text}"):

            col1, col2, col3 = st.columns(3)

            if entry_value >= 0:
                col1.success(f"Crédito entrada: +${entry_value:,.2f}")
            else:
                col1.error(f"Débito entrada: ${entry_value:,.2f}")

            col2.metric("Valor atual", f"${current_value:,.2f}")

            if total_pl >= 0:
                col3.success(f"P/L: +${total_pl:,.2f}")
            else:
                col3.error(f"P/L: ${total_pl:,.2f}")

            st.caption(f"IV média na entrada: {avg_iv*100:.1f}%")

            created_at_raw = strat.get("created_at")
            try:
                created_at_dt = datetime.fromisoformat(created_at_raw)
                created_at_formatted = created_at_dt.strftime("%d/%m/%Y às %H:%M")
            except:
                created_at_formatted = created_at_raw

            st.caption(f"Criada em: {created_at_formatted}")

            st.markdown("**Pernas:**")

            for leg in legs:

                side = leg.get("side", "").upper()
                opt_type = leg.get("type", "").upper()
                strike = leg.get("strike", 0)
                qty = float(leg.get("quantity", 0))
                exp_date = leg.get("expiration_date", "-")

                premium_usd = (
                    leg.get("premium_usd")
                    or leg.get("premium_entry_usd")
                    or 0
                )

                total_entry = float(premium_usd) * qty

                text = (
                    f"{side} {opt_type} | "
                    f"Strike {strike} | "
                    f"Qty {qty} | "
                    f"Exp: {exp_date}"
                )

                if leg.get("side") == "buy":
                    st.error(text + f" | Débito: -${total_entry:,.2f}")
                else:
                    st.success(text + f" | Crédito: +${total_entry:,.2f}")

except Exception as e:
    st.error(f"Erro ao carregar carteira: {e}")
