import streamlit as st

st.set_page_config(
    page_title="ETH Options Simulator",
    layout="wide"
)

# =========================
# HERO
# =========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.title("Simulador de Estratégias de Opções em ETH")

    st.markdown("""
Simule estratégias profissionais de opções de forma simples.

✔ Monte estratégias multi-perna  
✔ Visualize gráfico de payoff em tempo real  
✔ Acompanhe P/L da carteira  
✔ Dados de mercado da Deribit  

Ideal para traders de cripto que querem **controlar risco antes de operar**.
""")

    col_login, col_signup = st.columns(2)

    with col_login:
        if st.button("🔐 Fazer Login", use_container_width=True):
            st.switch_page("pages/1_Login.py")

    with col_signup:
        if st.button("🚀 Criar Conta", use_container_width=True):
            st.switch_page("pages/1_Login.py")


# =========================
# IMAGEM DO PAYOFF
# =========================
with col2:
    st.image(
        "https://i.imgur.com/3l7YQ0B.png",
        caption="Exemplo de gráfico de Payoff",
        use_container_width=True
    )


st.markdown("---")

# =========================
# FEATURES
# =========================
st.subheader("O que você pode fazer")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
### 📈 Simulação Avançada
- Call e Put  
- Estratégias multi-perna  
- Break-even automático  
- Probabilidade de lucro
""")

with f2:
    st.markdown("""
### 💼 Carteira
- P/L em tempo real  
- Valor de mercado das posições  
- Histórico de estratégias
""")

with f3:
    st.markdown("""
### ⚡ Dados em Tempo Real
- ETH Spot  
- IV de mercado  
- Prêmios da Deribit  
- Atualização automática
""")


st.markdown("---")

# =========================
# CALL TO ACTION FINAL
# =========================
st.markdown("## Comece agora gratuitamente")

col_center = st.columns([1,2,1])[1]

with col_center:
    if st.button("Criar conta e começar", use_container_width=True):
        st.switch_page("pages/1_Login.py")


st.markdown("---")

st.caption("ETH Options Simulator • MVP • Dados de mercado via Deribit")

