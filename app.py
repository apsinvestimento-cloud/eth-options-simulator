import streamlit as st

st.set_page_config(page_title="ETH Options Simulator", layout="wide")

# =========================
# AUTO REDIRECT SE LOGADO
# =========================
if "user" in st.session_state and st.session_state.user:
    st.switch_page("pages/2_simulador.py")

# =========================
# LANDING
# =========================

st.title("Simulador de Estratégias de Opções em ETH")

col1, col2 = st.columns([1.2, 1])

with col1:

    st.markdown("""
Simule estratégias profissionais de opções de forma simples.

✓ Monte estratégias multi-perna  
✓ Visualize gráfico de payoff em tempo real  
✓ Acompanhe P/L da carteira  
✓ Dados de mercado da Deribit  

Ideal para traders de cripto que querem **controlar risco antes de operar**.
""")

    st.markdown("###")

    col1, col2 = st.columns(2)

    with col1:
        st.link_button("🔐 Fazer Login", "/1_login")

    with col2:
        st.link_button("🚀 Ir para Simulador", "/2_simulador")


    with col2:
        st.image(
            "https://i.imgur.com/6X4QF4K.png",  # exemplo de gráfico payoff
            use_container_width=True
        )




