import streamlit as st

# =========================
# REQUER LOGIN
# =========================
def require_login():
    if "user" not in st.session_state or st.session_state.user is None:
        st.warning("Faça login na página Login")
        st.stop()

# =========================
# REQUER PLANO PRO
# =========================
def require_pro():
    require_login()
    
    if "role" not in st.session_state:
        st.warning("Sessão inválida. Faça login novamente.")
        st.stop()

    if st.session_state.role not in ["pro", "admin"]:
        st.error("Recurso disponível apenas para usuários PRO")
        st.stop()
