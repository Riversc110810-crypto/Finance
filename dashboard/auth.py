import hashlib
import streamlit as st


DEFAULT_USERNAME = "trader"
DEFAULT_PASSWORD  = "trading123"


def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def check_credentials(username: str, password: str, profile: dict) -> bool:
    stored_hash = profile.get("password_hash", hash_password(DEFAULT_PASSWORD))
    stored_user = profile.get("username", DEFAULT_USERNAME)
    return username.strip() == stored_user and hash_password(password) == stored_hash


def is_logged_in() -> bool:
    return st.session_state.get("authenticated", False)


def login_page(profile: dict):
    st.markdown("""
    <style>
      .login-wrap { max-width: 420px; margin: 8vh auto; }
      .login-title { font-size: 2.2rem; font-weight: 800; color: #00d4aa;
                     letter-spacing: -1px; margin-bottom: 0; }
      .login-sub   { color: #8892b0; margin-bottom: 2rem; font-size: 0.95rem; }
      .login-hint  { color: #555e7a; font-size: 0.78rem; margin-top: 1.5rem; text-align:center; }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">📈 Trading Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-sub">Personal hedge-fund style analytics platform</p>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="trader")
            password = st.text_input("Password", type="password", placeholder="••••••••••")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

        if submitted:
            if check_credentials(username, password, profile):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials.")

        st.markdown('<p class="login-hint">Default: trader / trading123 — change in Settings</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def logout():
    st.session_state["authenticated"] = False
    st.session_state.pop("chat_history", None)
    st.rerun()
