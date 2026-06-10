"""Frontend SPA inyectado (spec §7).

Templates HTML/JS/CSS premium fuera de app.py, inyectados vía iframes
same-origin de Streamlit. Theming exclusivamente por CSS variables del
documento padre (sin colores hardcodeados) y degradable si un CDN falla.
"""
from frontend.inject import inject_effects, render_bracket

__all__ = ["inject_effects", "render_bracket"]
