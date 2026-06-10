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
import logging

import streamlit as st

VIEWER = "viewer"
ADMIN = "admin"
_LOG = logging.getLogger("mundial.auth")


def verify_password(candidate: str, secret: str | None) -> bool:
    """Comparación en tiempo constante; False si no hay secret (fail-closed)."""
    if not secret or not candidate:
        return False
    return hmac.compare_digest(candidate.encode(), str(secret).encode())


def _secret_password() -> str | None:
    """Clave maestra desde st.secrets. Formato canónico (igual en local y
    en Streamlit Community Cloud): admin_password = "..." al tope del
    archivo. Se acepta [auth].admin_password por compatibilidad."""
    for getter in (lambda: st.secrets["admin_password"],
                   lambda: st.secrets["auth"]["admin_password"]):
        try:
            return getter()
        except (KeyError, FileNotFoundError):
            continue
    return None


def current_role() -> str:
    """Rol de la sesión actual (viewer por defecto)."""
    return st.session_state.get("role", VIEWER)


def is_admin() -> bool:
    return current_role() == ADMIN


@st.dialog("🔒 Acceso administrativo")
def _login_dialog() -> None:
    """Invariante R2 (no-leak): este flujo nunca muestra detalles de
    configuración (rutas de secrets, plantillas TOML) ni distingue entre
    clave incorrecta y auth sin configurar. Instrucciones de despliegue:
    ver CLAUDE.md, sección Despliegue."""
    if is_admin():
        st.markdown("Sesión activa como **Super User**.")
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state["role"] = VIEWER
            st.rerun()
        return
    st.caption("Las predicciones son públicas. El ingreso de "
               "resultados requiere clave de administrador.")
    with st.form("login", clear_on_submit=True, border=False):
        pwd = st.text_input("Clave de administrador", type="password")
        if st.form_submit_button("Entrar como admin",
                                 use_container_width=True):
            if verify_password(pwd, _secret_password()):
                _LOG.info("login admin OK")
                st.session_state["role"] = ADMIN
                st.rerun()
            else:
                # el motivo exacto va SOLO al log, nunca a pantalla
                _LOG.warning("login admin fallido (clave incorrecta o "
                             "auth sin configurar)")
                st.error("Acceso denegado.")


def login_entry() -> None:
    """Botón de login/logout para el header. Abre el modal de acceso
    (spec §8: NO usa st.sidebar — el header nativo de Streamlit está
    oculto por CSS y se llevaba consigo el control de re-expandirlo)."""
    label = "👑 Super User" if is_admin() else "🔒 Acceso admin"
    if st.button(label, key="login_entry", use_container_width=True):
        _login_dialog()
