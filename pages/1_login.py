import streamlit as st
from supabase import create_client

# =========================
# CONFIG SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Login", layout="centered")

st.title("🔐 Login / Conta")

# =========================
# SESSION INIT
# =========================
if "user" not in st.session_state:
    st.session_state.user = None

if "role" not in st.session_state:
    st.session_state.role = None


# =========================
# FUNÇÕES
# =========================

def get_user_profile(user_id):
    response = supabase.table("profiles") \
        .select("*") \
        .eq("id", user_id) \
        .execute()

    if response.data:
        return response.data[0]
    return None


def create_user_profile(user):
    supabase.table("profiles").insert({
        "id": user.id,
        "email": user.email,
        "role": "free"
    }).execute()


# =========================
# SE NÃO ESTIVER LOGADO
# =========================
if not st.session_state.user:

    email = st.text_input("Email")
    password = st.text_input("Senha", type="password")

    col1, col2 = st.columns(2)

    # ===== LOGIN =====
    with col1:
        if st.button("Entrar"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                user = response.user

                # Buscar perfil
                profile = get_user_profile(user.id)

                # Se não existir (usuário antigo)
                if not profile:
                    create_user_profile(user)
                    profile = get_user_profile(user.id)

                # Salvar sessão
                st.session_state.user = user
                st.session_state.role = profile["role"]

                st.success("Login realizado!")
                st.rerun()

            except Exception:
                st.error("Email ou senha inválidos")

    # ===== CADASTRO =====
    with col2:
        if st.button("Criar conta"):
            try:
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })

                user = response.user

                # Criar perfil FREE
                if user:
                    create_user_profile(user)

                st.success("Conta criada! Verifique seu email.")
            except Exception:
                st.error("Erro ao criar conta")


# =========================
# USUÁRIO LOGADO
# =========================
else:
    st.success(f"Logado como: {st.session_state.user.email}")

    role = st.session_state.role

    # Mostrar nível
    if role == "free":
        st.info("Plano: FREE")
    elif role == "pro":
        st.success("Plano: PRO")
    elif role == "admin":
        st.warning("ADMIN")

    st.markdown("---")

    # Logout
    if st.button("Sair"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()
