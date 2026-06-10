"""Inyección de los templates frontend en Streamlit.

Cada template se sirve dentro de un iframe same-origin
(`st.components.v1.html`), por lo que su JS puede leer las CSS variables
del documento padre y reaccionar al cambio de tema con un MutationObserver
(contrato de theming del spec §7).
"""
from __future__ import annotations

import json
from pathlib import Path
from string import Template

import streamlit.components.v1 as components

_TEMPLATES = Path(__file__).parent / "templates"


def _load(name: str) -> str:
    return (_TEMPLATES / name).read_text(encoding="utf-8")


def inject_effects() -> None:
    """Lenis (scroll suave) + GSAP (transiciones de tab) + Three.js
    (fondo liquid gradient reactivo al tema). Iframe invisible de 0 px."""
    components.html(_load("effects.html"), height=0, width=0)


def render_bracket(payload: dict, height: int = 780) -> None:
    """Bracket interactivo (spec §7: contrato JSON del bracket).

    payload = {"rounds": [...], "champion": {...}} — ver spec. El template
    resuelve layout con CSS Grid/Flexbox (sin matemáticas de márgenes) y
    anima el filtro de fases con GSAP.
    """
    html = Template(_load("bracket.html")).safe_substitute(
        DATA=json.dumps(payload, ensure_ascii=False))
    components.html(html, height=height, scrolling=False)
