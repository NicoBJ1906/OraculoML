"""RBAC ligero para la UI (spec §8).

Roles: ``viewer`` (default, solo lectura) y ``admin`` (ingesta en vivo).
La contraseña maestra vive en ``.streamlit/secrets.toml`` (gitignored):

    [auth]
    admin_password = "..."

Sin secrets configurados la app queda permanentemente en viewer
(fail-closed). Invariante R1: el tab de ingesta no se CONSTRUYE para
viewers, no solo se oculta.
"""
from __future__ import annotations

import hmac

import streamlit as st

VIEWER = "viewer"
ADMIN = "admin"


def verify_password(candidate: str, secret: str | None) -> bool:
    """Comparación en tiempo constante; False si no hay secret (fail-closed)."""
    if not secret or not candidate:
        return False
    return hmac.compare_digest(candidate.encode(), str(secret).encode())


def _secret_password() -> str | None:
    try:
        return st.secrets["auth"]["admin_password"]
    except (KeyError, FileNotFoundError):
        return None


def current_role() -> str:
    """Rol de la sesión actual (viewer por defecto)."""
    return st.session_state.get("role", VIEWER)


def is_admin() -> bool:
    return current_role() == ADMIN


def login_widget() -> None:
    """Caja de login/logout en el sidebar. Setea st.session_state['role']."""
    with st.sidebar:
        if is_admin():
            st.markdown("**Rol: Super User**")
            if st.button("Cerrar sesión", use_container_width=True):
                st.session_state["role"] = VIEWER
                st.rerun()
            return
        st.markdown("**Modo espectador**")
        st.caption("Las predicciones son públicas. El ingreso de "
                   "resultados requiere clave de administrador.")
        with st.form("login", clear_on_submit=True, border=False):
            pwd = st.text_input("Clave de administrador", type="password")
            if st.form_submit_button("Entrar como admin",
                                     use_container_width=True):
                if verify_password(pwd, _secret_password()):
                    st.session_state["role"] = ADMIN
                    st.rerun()
                else:
                    st.error("Clave incorrecta o auth no configurada.")
