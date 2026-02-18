import streamlit as st
from utils.auth import require_login

require_login()


st.set_page_config(page_title="Carteira", layout="wide")

st.title("📊 Carteira de Estratégias")

st.write("Página da carteira")
