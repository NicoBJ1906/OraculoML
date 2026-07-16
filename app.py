"""UI del Mundial 2026 — predicciones en vivo.

Temas light / dark (negro + rojo, fondo aurora animado). Ingesta extendida
(xG, goleadores, asistencias, tarjetas, lesiones, clima, formación) que
alimenta el LiveEngine: Feature State Updating + Online Learning sin
reentrenar el modelo base.

Uso:
    streamlit run app.py
"""
from __future__ import annotations

import html as html_lib
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ---- logging a logs/app.log (diagnóstico: cachés, sim, ingesta, login)
_LOGROOT = logging.getLogger("mundial")
if not _LOGROOT.handlers:
    (ROOT / "logs").mkdir(exist_ok=True)
    _h = RotatingFileHandler(ROOT / "logs" / "app.log", maxBytes=500_000,
                             backupCount=2, encoding="utf-8")
    _h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _LOGROOT.addHandler(_h)
    _LOGROOT.setLevel(logging.INFO)
LOG = logging.getLogger("mundial.app")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from frontend import inject_effects, render_bracket
from mundial import auth
from mundial.display import display_pred
from mundial.live.engine import LiveEngine
from mundial.live.store import LiveStore
from mundial.predict.engine import MARKET_WEIGHT, devig
from mundial.predict.montecarlo import TournamentSimulator

def _github_cfg() -> tuple[str | None, str, str]:
    """Lee configuración GitHub desde st.secrets (falla silenciosamente)."""
    try:
        token = str(st.secrets["github_token"])
        repo = str(st.secrets.get("github_repo", "NicoBJ1906/OraculoML"))
        branch = str(st.secrets.get("github_branch", "main"))
        return token, repo, branch
    except Exception:  # noqa: BLE001 — sin secrets o en tests bare
        return None, "NicoBJ1906/OraculoML", "main"


_gh_token, _gh_repo, _gh_branch = _github_cfg()
STORE = LiveStore(ROOT, github_token=_gh_token, github_repo=_gh_repo,
                  github_branch=_gh_branch)
HOSTS = {"United States", "Mexico", "Canada"}
esc = html_lib.escape

EVENT_LABELS = {"Gol": "goal", "Penal": "penalty", "Autogol": "own_goal",
                "Asistencia": "assist"}
CARD_LABELS = {"Amarilla": "yellow", "Roja": "red"}
INJURY_LABELS = {"Baja un partido": "next_match",
                 "Baja resto del torneo": "tournament"}
WEATHER_OPTS = ["Sin dato", "Despejado", "Calor extremo", "Lluvia",
                "Tormenta", "Frío"]
OTRO = "✏️ Otro…"          # escape del dropdown de jugadores a texto libre

# Códigos ISO para banderas vía flagcdn.com (los emojis de bandera no
# renderizan en Windows).
FLAG_ISO = {
    "Albania": "al", "Algeria": "dz", "Argentina": "ar", "Australia": "au",
    "Austria": "at", "Belgium": "be", "Bolivia": "bo",
    "Bosnia and Herzegovina": "ba", "Brazil": "br",
    "Canada": "ca", "Cape Verde": "cv", "Colombia": "co", "Croatia": "hr",
    "Curaçao": "cw", "Czech Republic": "cz", "Denmark": "dk", "DR Congo": "cd",
    "Ecuador": "ec", "Egypt": "eg", "England": "gb-eng", "France": "fr",
    "Germany": "de", "Ghana": "gh", "Haiti": "ht", "Iran": "ir",
    "Iraq": "iq", "Italy": "it", "Ivory Coast": "ci", "Jamaica": "jm",
    "Japan": "jp", "Jordan": "jo", "Kosovo": "xk",
    "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl",
    "New Caledonia": "nc", "New Zealand": "nz", "North Macedonia": "mk",
    "Northern Ireland": "gb-nir", "Norway": "no", "Panama": "pa",
    "Paraguay": "py", "Poland": "pl", "Portugal": "pt", "Qatar": "qa",
    "Republic of Ireland": "ie", "Romania": "ro", "Saudi Arabia": "sa",
    "Scotland": "gb-sct", "Senegal": "sn", "Slovakia": "sk",
    "South Africa": "za", "South Korea": "kr", "Spain": "es",
    "Suriname": "sr", "Sweden": "se", "Switzerland": "ch", "Tunisia": "tn",
    "Turkey": "tr", "Ukraine": "ua", "United States": "us", "Uruguay": "uy",
    "Uzbekistan": "uz", "Wales": "gb-wls",
}

_FAVICON = ROOT / "assets" / "favicon.png"
st.set_page_config(page_title="Oráculo personal de Nicolás — Mundial 2026",
                   page_icon=str(_FAVICON) if _FAVICON.exists() else "⚽",
                   layout="wide")

# ----------------------------------------------------------------- temas
PALETTES = {
    "dark": {
        "bg": "#070708", "surface": "rgba(255,255,255,.045)",
        "surface-solid": "#121214", "border": "rgba(255,255,255,.10)",
        "text": "#f5f5f7", "muted": "#9a9aa3",
        "accent": "#ff2d55", "accent2": "#ff6b3d",
        "bar-d": "rgba(255,255,255,.18)",
        "bar-a1": "#0a84ff", "bar-a2": "#64d2ff",
        "glow": "rgba(255,45,85,.20)", "shadow": "rgba(0,0,0,.5)",
        "input-bg": "rgba(255,255,255,.07)",
        "thead": "rgba(255,255,255,.05)",
        "blob1": "#e11d48", "blob2": "#7f1d1d", "blob3": "#4c0519",
        "blob-op": ".42", "hero-sub": "#9a9aa3",
    },
    "light": {
        "bg": "#f6f6f8", "surface": "#ffffff",
        "surface-solid": "#ffffff", "border": "rgba(0,0,0,.09)",
        "text": "#1d1d1f", "muted": "#6e6e73",
        "accent": "#e11d48", "accent2": "#f97316",
        "bar-d": "rgba(0,0,0,.14)",
        "bar-a1": "#2563eb", "bar-a2": "#60a5fa",
        "glow": "rgba(225,29,72,.10)", "shadow": "rgba(0,0,0,.08)",
        "input-bg": "#ffffff",
        "thead": "rgba(0,0,0,.035)",
        "blob1": "#fda4af", "blob2": "#fecdd3", "blob3": "#e0e7ff",
        "blob-op": ".30", "hero-sub": "#6e6e73",
    },
}

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, p, span, div, label, button, input, select, textarea {
  font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp {background: var(--bg);}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.block-container {padding-top: 2.2rem; max-width: 1280px;}
section[data-testid="stMain"], .block-container {position: relative; z-index: 1;}

/* ---- scrollbar ---- */
::-webkit-scrollbar {width: 6px; height: 6px;}
::-webkit-scrollbar-track {background: transparent;}
::-webkit-scrollbar-thumb {background: var(--border); border-radius: 999px;}
::-webkit-scrollbar-thumb:hover {background: var(--muted);}

/* ---- fondo aurora animado ---- */
.bg-blobs {position: fixed; inset: 0; z-index: 0; overflow: hidden;
           pointer-events: none;}
.blob {position: absolute; border-radius: 50%; filter: blur(130px);
       opacity: var(--blob-op); will-change: transform;}
.b1 {width: 620px; height: 620px; top: -180px; left: -140px;
     background: radial-gradient(circle, var(--blob1), transparent 70%);
     animation: drift1 26s ease-in-out infinite alternate;}
.b2 {width: 700px; height: 700px; top: -80px; right: -200px;
     background: radial-gradient(circle, var(--blob2), transparent 70%);
     animation: drift2 32s ease-in-out infinite alternate;}
.b3 {width: 560px; height: 560px; bottom: -240px; left: 30%;
     background: radial-gradient(circle, var(--blob3), transparent 70%);
     animation: drift3 38s ease-in-out infinite alternate;}
@keyframes drift1 {to {transform: translate(150px, 120px) scale(1.2);}}
@keyframes drift2 {to {transform: translate(-130px, 100px) scale(1.12);}}
@keyframes drift3 {to {transform: translate(110px, -130px) scale(1.25);}}

/* ---- sidebar: ya no se usa (login via modal, spec §8), pero si algo
   llegara a renderizar en él, el control nativo de expandir/colapsar
   debe seguir visible aunque el header esté oculto ---- */
div[data-testid="stSidebarCollapsedControl"],
span[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] {
  visibility: visible !important; position: fixed; top: 10px; left: 10px;
  z-index: 999; color: var(--text);}

/* ---- hero ---- */
.hero {
  font-size: 3.2rem; font-weight: 800; letter-spacing: -.03em;
  line-height: 1.08; color: var(--text); margin: 0 0 .35rem 0;}
.hero .grad {background: linear-gradient(92deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero-sub {color: var(--hero-sub); font-size: .88rem; margin: 0 0 .4rem 0;
           font-weight: 400;}
h3, h2 {font-weight: 700 !important; letter-spacing: -.02em !important;
        color: var(--text) !important;}
a[href^="#"] {display: none !important;}
p, .stMarkdown, label, .stCaption, div[data-testid="stCaptionContainer"],
div[data-testid="stWidgetLabel"] p {color: var(--text);}
div[data-testid="stCaptionContainer"], .stCaption p {color: var(--muted) !important;}
hr {border-color: var(--border) !important;}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  backdrop-filter: blur(24px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 999px; padding: 5px; gap: 2px; width: fit-content;
  box-shadow: 0 8px 36px var(--glow);
  transition: box-shadow .4s ease;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 999px; padding: 7px 20px; font-weight: 600; font-size: .82rem;
  background: transparent; color: var(--muted);
  transition: all .25s ease;
}
.stTabs [data-baseweb="tab"]:hover {color: var(--text); background: rgba(255,255,255,.04);}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #fff !important;
  box-shadow: 0 4px 20px var(--glow);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]
  {display: none;}

/* ---- cards ---- */
.glass {
  background: var(--surface);
  backdrop-filter: blur(28px) saturate(160%);
  -webkit-backdrop-filter: blur(28px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: 0 12px 44px var(--shadow), 0 0 50px var(--glow);
  padding: 18px 20px; margin-bottom: 14px;
  transition: transform .3s cubic-bezier(.16,1,.3,1), box-shadow .3s ease, border-color .3s ease;
}
.glass:hover {
  transform: translateY(-3px);
  border-color: var(--accent);
  box-shadow: 0 18px 54px var(--shadow), 0 0 60px var(--glow);
}
.match-card {min-height: 160px; cursor: pointer; position: relative;}
.match-card:active {transform: scale(.98);}
.mc-meta {font-size: .65rem; font-weight: 600; letter-spacing: .1em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 10px;}
.mc-row {display: flex; align-items: center; justify-content: space-between; gap: 8px;}
.mc-team {flex: 1; text-align: center; font-weight: 600; font-size: .85rem;
          color: var(--text); line-height: 1.2;}
.mc-team .flag {display: block; margin-bottom: 4px;}
/* tamaño ESTRICTO: los PNG de flagcdn traen alturas distintas por país
   y descuadran las cards — se normalizan a 40x26 recortando */
.mc-team .flag img {width: 40px; height: 26px; object-fit: cover;
  border-radius: 4px; filter: drop-shadow(0 4px 12px var(--shadow));}
.mc-probs-wrap {display: flex; justify-content: space-between; align-items: baseline;
                margin-top: 12px; gap: 6px;}
.mc-prob-block {text-align: center; flex: 1;}
.mc-prob-block .val {font-size: 1.15rem; font-weight: 800; letter-spacing: -.02em;
                     display: block; line-height: 1.1;}
.mc-prob-block .lbl {font-size: .6rem; font-weight: 600; letter-spacing: .08em;
                     text-transform: uppercase; color: var(--muted); display: block;}
.mc-prob-block.h .val {color: var(--accent);}
.mc-prob-block.d .val {color: var(--text);}
.mc-prob-block.a .val {color: var(--bar-a1);}

/* ---- podio Camino al título ---- */
.mc-probs {display: flex; font-size: .78rem; font-weight: 600; color: var(--muted);}
.mc-probs span:first-child {color: var(--accent);}
.mc-probs span:last-child {color: var(--bar-a1);}
.podium-pct {font-size: 2rem; font-weight: 800; letter-spacing: -.02em;
  line-height: 1.15; margin-top: 4px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;}
.podium-lbl {font-size: .6rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 10px;}
.mc-btn-xai {position: absolute; top: 10px; right: 12px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 999px; width: 28px; height: 28px; display: flex;
  align-items: center; justify-content: center; font-size: .7rem;
  color: var(--muted); cursor: pointer;
  transition: all .25s ease; padding: 0; line-height: 1;}
.mc-btn-xai:hover {background: var(--input-bg); border-color: var(--accent); color: var(--accent);}
.pill {display: inline-block; background: var(--surface);
       border-radius: 999px; padding: 4px 13px; margin: 2px 5px 2px 0;
       font-size: .8rem; font-weight: 600;
       border: 1px solid var(--border); color: var(--text);
       transition: all .2s ease;}
.pill:hover {border-color: var(--accent);}
.mc-adj {display: inline-block; margin-left: 8px; padding: 1px 8px;
         border-radius: 999px; background: var(--surface);
         border: 1px solid var(--accent); color: var(--accent);
         font-size: .6rem; letter-spacing: .06em;}
.rowchip {background: var(--input-bg); border: 1px solid var(--border);
          border-radius: 10px; padding: 7px 12px; font-size: .84rem;
          color: var(--text); overflow: hidden; text-overflow: ellipsis;
          white-space: nowrap;}

/* ---- XAI modal ---- */
div[data-testid="stDialog"] {background: transparent !important;}
div[data-testid="stDialog"] > div {
  background: color-mix(in srgb, var(--surface-solid) 92%, transparent) !important;
  backdrop-filter: blur(40px) saturate(160%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 28px !important;
  box-shadow: 0 24px 80px var(--shadow), 0 0 80px var(--glow) !important;
  padding: 28px !important;
}
div[data-testid="stDialog"] div[data-testid="stMarkdownContainer"] p {color: var(--text);}
.xai-team {font-weight: 800; font-size: 1.1rem; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;}
.xai-stat {display: flex; justify-content: space-between; padding: 6px 0;
           border-bottom: 1px solid var(--border); font-size: .85rem;}
.xai-stat:last-child {border-bottom: none;}
.xai-stat .label {color: var(--muted);}
.xai-stat .value {font-weight: 700;}
.xai-stat.pos .value {color: var(--accent2);}
.xai-stat.neg .value {color: var(--bar-a1);}
.xai-divider {height: 1px; background: var(--border); margin: 8px 0;}

/* ---- slider de horizonte: separación de las tarjetas ---- */
div[data-testid="stSlider"] {margin-bottom: 32px; padding-top: 34px;}
div[data-testid="stSliderThumbValue"] {
  white-space: nowrap; min-width: 30px; text-align: center;
  font-size: .8rem !important; font-weight: 700; color: #fff !important;
  background: var(--accent); padding: 2px 8px; border-radius: 999px;
  box-shadow: 0 2px 8px var(--glow);}

/* ---- centrado de botones nativos (Explicar pronóstico) ---- */
div[data-testid="stButton"] {display: flex; justify-content: center;
  width: 100%; margin-top: 10px;}

/* ---- modal XAI: tema nativo forzado a las vars ---- */
div[data-testid="stDialog"] > div, div[role="dialog"] {
  background: var(--surface-solid) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important;}
div[data-testid="stDialog"] p, div[data-testid="stDialog"] span,
div[data-testid="stDialog"] label, div[role="dialog"] h1,
div[role="dialog"] h2, div[role="dialog"] h3 {color: var(--text) !important;}
/* baseweb renderiza el modal en un portal con el tema base (dark de
   config.toml): forzar TODA la cadena a las vars para que el modal siga
   el tema activo y no "se pase a dark" en modo claro */
div[data-baseweb="modal"] > div, div[aria-modal="true"],
div[data-baseweb="modal"] section {
  background: var(--surface-solid) !important; color: var(--text) !important;}
.katex, .katex .mord, .katex .mrel {color: var(--text) !important;}

/* ---- modal de login: input y botón integrados en ambos temas ---- */
div[data-testid="stDialog"] .stTextInput input {
  background: var(--input-bg) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important;}
div[data-testid="stDialog"] .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--glow) !important;}
div[data-testid="stDialog"] .stTextInput button,
div[data-testid="stDialog"] .stTextInput svg {
  color: var(--muted) !important; fill: var(--muted) !important;}
div[data-testid="stDialog"] [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;}
div[data-testid="stDialog"] div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #fff !important; border: none !important;
  box-shadow: 0 6px 22px var(--glow);}
div[data-testid="stDialog"] > div > button svg {fill: var(--muted);}

/* ---- date picker (calendario de st.date_input, tab Eliminatorias):
   sin colores quemados — solo variables del tema ---- */
div[data-baseweb="calendar"], div[data-baseweb="datepicker"] {
  background: var(--surface-solid) !important;
  border-radius: 16px !important; color: var(--text) !important;}
div[data-baseweb="calendar"] div, div[data-baseweb="calendar"] span,
div[data-baseweb="calendar"] button {
  color: var(--text) !important; background: transparent;}
div[data-baseweb="calendar"] svg {fill: var(--text) !important;}
div[data-baseweb="calendar"] [aria-selected="true"],
div[data-baseweb="calendar"] [aria-selected="true"] > div {
  background: var(--accent) !important; color: #fff !important;
  border-radius: 999px;}
div[data-baseweb="calendar"] [role="gridcell"]:hover > div {
  background: color-mix(in srgb, var(--accent) 18%, transparent) !important;
  border-radius: 999px;}
div[data-baseweb="popover"]:has([data-baseweb="calendar"]) > div {
  background: var(--surface-solid) !important;
  border: 1px solid var(--border) !important; border-radius: 16px !important;
  box-shadow: 0 16px 50px var(--shadow) !important;}

/* ---- inputs y botones ---- */
.stButton > button {
  border-radius: 999px; font-weight: 700; border: none;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff; padding: .5rem 1.5rem;
  box-shadow: 0 6px 22px var(--glow);
  transition: transform .25s cubic-bezier(.16,1,.3,1), box-shadow .3s ease;
}
.stButton > button:hover {transform: translateY(-2px) scale(1.02); color: #fff; box-shadow: 0 10px 32px var(--glow);}
.stButton > button:active {transform: scale(.97);}
div[data-testid="stFormSubmitButton"] button {
  border-radius: 999px; font-weight: 700; border: 1px solid var(--border);
  background: var(--input-bg); color: var(--text); width: 100%;}
div[data-testid="stFormSubmitButton"] button:hover {
  border-color: var(--accent); color: var(--accent);}
div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input,
.stTextInput input, .stTextArea textarea {
  border-radius: 14px !important; background: var(--input-bg) !important;
  color: var(--text) !important; border-color: var(--border) !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-baseweb="select"] > div:focus-within, .stNumberInput input:focus,
.stDateInput input:focus, .stTextInput input:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--glow) !important;
}
div[data-baseweb="select"] svg {fill: var(--muted);}
div[data-baseweb="popover"] ul, ul[role="listbox"] {
  background: var(--surface-solid) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;}
ul[role="listbox"] li {color: var(--text) !important;
  transition: background .15s ease;}
ul[role="listbox"] li:hover {background: var(--input-bg) !important;}
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px; padding: 14px 18px;
  box-shadow: 0 12px 40px var(--shadow);
  transition: transform .25s ease, box-shadow .25s ease;
}
div[data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 16px 48px var(--shadow);}
div[data-testid="stMetricValue"] {
  font-weight: 700;
  color: var(--text);}
div[data-testid="stMetricLabel"] p {color: var(--muted) !important;}
div[data-testid="stExpander"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px;
  transition: border-color .2s ease;}
div[data-testid="stExpander"]:hover {border-color: color-mix(in srgb, var(--accent) 30%, var(--border));}
div[data-testid="stExpander"] summary {color: var(--text); font-weight: 600; padding: 8px 0;}
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {padding: 4px 0 8px 0;}
/* los editores de filas (goles/tarjetas/lesiones) viven en columnas
   angostas dentro del expander: sin min-width los inputs se desbordan
   del contenedor glass */
div[data-testid="stExpanderDetails"] {overflow: hidden;}
div[data-testid="stExpanderDetails"] div[data-testid="stColumn"],
div[data-testid="stExpanderDetails"] div[data-testid="column"] {min-width: 0;}
div[data-testid="stExpanderDetails"] .stNumberInput input,
div[data-testid="stExpanderDetails"] .stTextInput input {min-width: 0;
  padding-left: 8px; padding-right: 4px;}
div[data-testid="stExpanderDetails"] div[data-baseweb="select"] > div
  {min-width: 0;}
div[data-testid="stExpanderDetails"] div[data-testid="stButton"]
  {margin-top: 0;}
/* legibilidad de selectboxes en el formulario admin: texto y opciones
   más grandes (antes quedaban ilegibles en columnas angostas) */
div[data-baseweb="select"] > div {font-size: .92rem; min-height: 42px;}
ul[role="listbox"] li {font-size: .92rem; padding-top: 8px !important;
  padding-bottom: 8px !important;}
div[data-testid="stSlider"] p {color: var(--muted); font-weight: 500; font-size: .82rem;}
div[data-testid="stSlider"] div[data-baseweb="slider"] {height: 6px !important;}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  height: 6px !important;
  border-radius: 999px !important;
}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  border-radius: 14px; overflow: hidden;
  border: 1px solid var(--border);}

/* ---- sliders personalizados ---- */
div[data-testid="stSlider"] div[role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--bg) !important;
  width: 20px !important;
  height: 20px !important;
  box-shadow: 0 0 0 4px var(--glow), 0 4px 12px var(--shadow) !important;
  transition: box-shadow .2s ease, transform .15s ease !important;
}
div[data-testid="stSlider"] div[role="slider"]:hover {
  transform: scale(1.15);
  box-shadow: 0 0 0 6px var(--glow), 0 6px 20px var(--shadow) !important;
}

/* ---- tablas HTML ---- */
.tblwrap {background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; overflow: auto; margin-bottom: 14px;
  box-shadow: 0 10px 36px var(--shadow);
  transition: box-shadow .3s ease;}
.tblwrap:hover {box-shadow: 0 14px 44px var(--shadow);}
.tbl {width: 100%; border-collapse: collapse; font-size: .82rem;}
.tbl th {position: sticky; top: 0; background: var(--thead);
  backdrop-filter: blur(20px);
  color: var(--muted); text-transform: uppercase; font-size: .64rem;
  letter-spacing: .1em; font-weight: 600; text-align: left;
  padding: 10px 14px; border-bottom: 1px solid var(--border);}
.tbl td {padding: 8px 14px; color: var(--text);
  border-bottom: 1px solid var(--border); white-space: nowrap;}
.tbl tr:last-child td {border-bottom: none;}
.tbl .t-team {font-weight: 600;}
.tbl .t-team img {vertical-align: -3px; margin-right: 7px; border-radius: 3px;}
.tbl .t-bar {min-width: 130px;}
.pbar {display: inline-block; width: 80px; height: 6px; border-radius: 999px;
  background: var(--bar-d); overflow: hidden; vertical-align: 2px;
  margin-right: 8px;
  transition: width .6s ease;}
.pbar div {height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));}

/* ---- leader cards ---- */
.leader-card {
  background: linear-gradient(135deg, var(--surface), color-mix(in srgb, var(--accent) 6%, var(--surface)));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  text-align: center;
  box-shadow: 0 8px 30px var(--shadow);
  transition: transform .3s cubic-bezier(.16,1,.3,1), box-shadow .3s ease, border-color .3s ease;
}
.leader-card:hover {
  transform: translateY(-4px) scale(1.02);
  border-color: var(--accent);
  box-shadow: 0 14px 44px var(--shadow), 0 0 40px var(--glow);
}
.leader-num {font-size: 1.6rem; font-weight: 900; letter-spacing: -.04em;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; line-height: 1;}
.leader-name {font-weight: 700; font-size: .92rem; color: var(--text); margin-top: 2px;}
.leader-team {font-size: .72rem; font-weight: 600; color: var(--muted);}
.leader-stat {font-size: .78rem; font-weight: 700; color: var(--accent2); margin-top: 6px;}
.leader-badge {display: inline-block; font-size: .55rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
  background: var(--input-bg); border-radius: 999px; padding: 1px 10px;
  margin-top: 6px;}

/* ---- form cards ---- */
.form-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 20px 22px;
  box-shadow: 0 10px 38px var(--shadow);
  transition: border-color .25s ease, box-shadow .25s ease;
}
.form-card:hover {border-color: color-mix(in srgb, var(--accent) 40%, var(--border));}
.form-card h4 {font-size: .85rem; font-weight: 700; color: var(--text); margin: 0 0 12px 0;
               letter-spacing: -.01em;}

/* ---- tablas de grupos ---- */
.group-card {margin-bottom: 18px;}
.group-card h4 {font-size: .85rem; font-weight: 700; color: var(--text); margin: 0 0 6px 0;}

/* ---- info boxes ---- */
.stAlert {border-radius: 16px !important; border: 1px solid var(--border) !important;}
div[data-testid="stInfo"] {background: var(--surface) !important; color: var(--text) !important;}
div[data-testid="stSuccess"] {background: var(--surface) !important; color: var(--accent2) !important;}
div[data-testid="stError"] {background: var(--surface) !important;}

/* ---- sin hint "Press Enter to submit form" en inputs ---- */
div[data-testid="InputInstructions"],
span[data-testid="InputInstructions"] {display: none !important;}

/* ---- checkboxes / toggles ---- */
div[data-testid="stCheckbox"] label span {font-weight: 600;}
.stToggle {gap: 8px;}

/* ---- íconos Material de Streamlit: NO heredan Poppins. Sin esto el
   navegador muestra el nombre del glifo como texto plano
   ("keyboard_arrow_down", "arrow_right") en expanders y selects ---- */
span[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded' !important;
  font-weight: normal !important; line-height: 1 !important;}

/* ---- paneles nativos con borde (tab admin): recuadro REAL que envuelve
   los widgets — responsive, sin divs huérfanos ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 22px !important;
  box-shadow: 0 10px 38px var(--shadow);
  transition: border-color .25s ease;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: color-mix(in srgb, var(--accent) 40%, var(--border)) !important;}
.fc-title {font-size: .85rem; font-weight: 700; color: var(--text);
  margin: 0 0 4px 0; letter-spacing: -.01em;}

/* ---- pronóstico mostrado en la card (cosmético) ---- */
.mc-pred {margin-top: 10px; text-align: center; font-weight: 700;
  font-size: .82rem; padding: 6px 10px; border-radius: 12px;
  border: 1px solid var(--border);}
.mc-pred.h {color: var(--accent); border-color: var(--accent);}
.mc-pred.a {color: #19c8ff; border-color: #19c8ff;}
.mc-pred.d {color: var(--muted);}

/* ---- tab Eliminatorias: cards de marcadores más probables ----
   grid auto-fit: 5 columnas en desktop, 2-3 en móvil sin media queries */
.sl-grid {display: grid; gap: 10px; margin: 12px 0 4px;
  grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));}
.sl-card {background: var(--input-bg); border: 1px solid var(--border);
  border-radius: 16px; padding: 12px 8px; text-align: center;
  transition: transform .2s ease, border-color .2s ease;}
.sl-card:hover {transform: translateY(-3px); border-color: var(--accent);}
.sl-card.top {background: linear-gradient(135deg, var(--accent), var(--accent2));
  border: none; box-shadow: 0 8px 24px var(--glow);}
.sl-tag {font-size: .56rem; letter-spacing: .12em; font-weight: 700;
  color: var(--muted); text-transform: uppercase;}
.sl-score {font-size: 1.35rem; font-weight: 800; color: var(--text);
  margin-top: 2px;}
.sl-pct {font-weight: 700; color: var(--accent); font-size: .95rem;}
.sl-card.top .sl-tag {color: rgba(255,255,255,.85);}
.sl-card.top .sl-score, .sl-card.top .sl-pct {color: #fff;}
.sl-bar {height: 4px; border-radius: 99px; background: var(--border);
  margin-top: 8px; overflow: hidden;}
.sl-bar div {height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));}
.sl-card.top .sl-bar {background: rgba(255,255,255,.3);}
.sl-card.top .sl-bar div {background: #fff;}
.bet-hint {margin-top: 12px; padding: 10px 14px; border-radius: 14px;
  border: 1px dashed var(--accent); font-size: .85rem; color: var(--text);}

/* ---- tab Auditoría: filas de backtesting ---- */
.audit-row {display: flex; align-items: center; gap: 16px;
  padding: 14px 18px; margin-bottom: 10px;}
.audit-row .ar-date {font-size: .66rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  min-width: 92px;}
.audit-row .ar-rival {flex: 1; font-weight: 700; color: var(--text);
  display: flex; align-items: center; gap: 8px; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.audit-row .ar-rival img {width: 26px; height: 17px; object-fit: cover;
  border-radius: 3px; flex: none;}
.audit-row .ar-cond {font-size: .6rem; font-weight: 700; color: var(--muted);
  letter-spacing: .1em;}
.audit-row .ar-prob {font-weight: 800; font-size: 1.05rem;
  color: var(--accent); min-width: 110px; text-align: right;}
.audit-row .ar-prob small {display: block; font-size: .58rem;
  font-weight: 600; letter-spacing: .08em; color: var(--muted);
  text-transform: uppercase;}
.audit-row .ar-score {font-weight: 800; font-size: 1.05rem;
  color: var(--text); min-width: 64px; text-align: center;}
.audit-row .ar-hit {font-size: 1.3rem; min-width: 36px; text-align: center;}

/* ================= RESPONSIVE MÓVIL (invariante U5 del spec) =================
   Único bloque móvil de toda la app — el desktop queda intacto por diseño.
   Regla de oro: la PÁGINA nunca scrollea en X; solo tab-list, .tblwrap y
   el bracket (dentro de su iframe) tienen scroll-x propio. */
@media (max-width: 768px) {
  /* -- corte global del desbordamiento fantasma -- */
  .stApp, section[data-testid="stMain"],
  div[data-testid="stMainBlockContainer"], .block-container {
    max-width: 100vw !important; overflow-x: hidden !important;
    box-sizing: border-box;}
  .block-container {padding: 1.3rem .9rem 3rem .9rem;}

  /* -- hero compacto -- */
  .hero {font-size: 1.9rem;}
  .hero-sub {font-size: .76rem;}

  /* -- tab pill: 7 tabs > viewport → scroll-x interno, scrollbar oculta -- */
  .stTabs [data-baseweb="tab-list"] {
    width: 100%; max-width: 100%; overflow-x: auto;
    flex-wrap: nowrap; scrollbar-width: none;
    -webkit-overflow-scrolling: touch;}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {display: none;}
  .stTabs [data-baseweb="tab"] {padding: 7px 13px; font-size: .74rem;
    white-space: nowrap; flex: none;}

  /* -- cards: padding y tipografía a escala de bolsillo -- */
  .glass {padding: 14px; border-radius: 18px;}
  .match-card {min-height: 0;}
  .mc-team {font-size: .76rem;}
  .mc-team .flag img {width: 34px; height: 22px;}
  .mc-prob-block .val {font-size: 1rem;}
  .podium-pct {font-size: 1.5rem;}
  .form-card {padding: 14px;}

  /* -- auditoría: la suma de min-widths fijos rompe <420px → wrap -- */
  .audit-row {flex-wrap: wrap; gap: 4px 12px; padding: 12px 14px;}
  .audit-row .ar-rival {flex: 1 1 100%; white-space: normal;}
  .audit-row .ar-date, .audit-row .ar-prob, .audit-row .ar-score,
  .audit-row .ar-hit {min-width: 0 !important;}
  .audit-row .ar-prob {text-align: left;}

  /* -- modales a ancho completo del viewport móvil -- */
  div[data-testid="stDialog"] > div {padding: 18px !important;
    border-radius: 20px !important;}

  /* -- blobs: blur de 130px castiga GPUs móviles -- */
  .blob {filter: blur(80px);}
}
</style>
"""

AURORA_HTML = ('<div class="bg-blobs"><div class="blob b1"></div>'
               '<div class="blob b2"></div><div class="blob b3"></div></div>')


def inject_theme() -> None:
    light = st.session_state.get("light_mode", False)
    pal = PALETTES["light" if light else "dark"]
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in pal.items()) + "}"
    st.markdown(f"<style>{root}</style>{BASE_CSS}{AURORA_HTML}",
                unsafe_allow_html=True)


inject_theme()
inject_effects()          # Lenis + GSAP + fondo WebGL (degradable, spec §7)


# --------------------------------------------- puerta de entrada (viewers)
def _club_gate() -> None:
    """Login de acceso general "El club de amigos de Nico": solo entra
    quien tenga la clave (secret `club_password`). Independiente del
    admin (auth.py, intacto). Sin el secret configurado no hay puerta
    (local / tests / CI siguen igual)."""
    try:
        clave = str(st.secrets["club_password"])
    except Exception:
        return
    if st.session_state.get("club_ok"):
        return
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        import base64
        foto = ROOT / "assets" / "club.jpeg"
        img = ('<img src="data:image/jpeg;base64,'
               + base64.b64encode(foto.read_bytes()).decode()
               + '" style="width:100%;max-width:380px;border-radius:18px;'
               'box-shadow:0 10px 32px var(--shadow)">'
               ) if foto.exists() else '<div style="font-size:2.6rem">⚽</div>'
        st.markdown(
            '<div class="glass" style="text-align:center;margin-top:6vh;'
            f'padding:30px 26px">{img}'
            '<h2 class="hero" style="font-size:1.7rem;margin:14px 0 4px">'
            'El oráculo personal de Nicolás</h2>'
            '<p style="color:var(--muted);font-size:.9rem;margin-bottom:0">'
            'Solo para mis amiguitos — solo quien me conoce sabe la '
            'clave.</p></div>',
            unsafe_allow_html=True)
        tries = st.session_state.get("club_tries", 0)
        if tries >= 3:
            st.error("🚫 Tres intentos fallidos — ya no puedes entrar. "
                     "Habla con Nicolás si crees que mereces otra "
                     "oportunidad.")
            st.stop()
        pw = st.text_input("Clave del club", type="password",
                           key="club_pw", label_visibility="collapsed",
                           placeholder="🔮 Clave para ver el futuroooo")
        if st.button("📻 Mami prenda la radio, encienda la tele 📺", type="primary",
                     use_container_width=True):
            if pw == clave:
                st.session_state["club_ok"] = True
                st.rerun()
            st.session_state["club_tries"] = tries + 1
            left = 2 - tries
            st.error("¡Esa no es la clave mi crack! Keep trying 😎"
                     + (f" (te quedan {left} intentos)" if left > 0
                        else " (último intento agotado)"))
            if tries + 1 >= 3:
                st.rerun()
        qr = ROOT / "assets" / "qr.png"
        if qr.exists():
            st.markdown(
                '<div style="text-align:center;margin-top:18px">'
                '<p style="color:var(--muted);font-size:.85rem;'
                'margin-bottom:8px">¿Quejas? Escanea este QR 👇</p>'
                '<img src="data:image/png;base64,'
                + base64.b64encode(qr.read_bytes()).decode()
                + '" style="width:130px;border-radius:12px;background:#fff;'
                'padding:6px"></div>', unsafe_allow_html=True)
    st.stop()


_club_gate()


# ----------------------------------------------------------------- carga
@st.cache_resource
def load_artifacts() -> dict:
    art = joblib.load(ROOT / "models" / "artifacts.joblib")
    LOG.info("artifacts cargados: %s partidos hasta %s",
             art.get("n_train"), art.get("trained_until"))
    return art


@st.cache_data
def load_teams() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "interim" / "teams_2026.parquet")


@st.cache_data
def load_fixtures() -> pd.DataFrame:
    """Fixtures de fase de grupos (sin resultado) desde results.csv."""
    df = pd.read_csv(ROOT / "data" / "raw" / "international" / "results.csv",
                     parse_dates=["date"])
    fx = df[df.home_score.isna() & (df.tournament == "FIFA World Cup")
            & (df.date.dt.year == 2026)].copy()
    fx["neutral"] = fx["neutral"].astype("string").str.upper().isin(["TRUE", "1"])
    groups = load_teams().set_index("name_canonical")["group"]
    fx["group"] = fx["home_team"].map(groups)
    return fx[["date", "home_team", "away_team", "city", "country",
               "neutral", "group"]].sort_values("date").reset_index(drop=True)


@st.cache_data
def load_ko_raw() -> list[dict]:
    wc = json.loads((ROOT / "data" / "raw" / "worldcup2026" / "worldcup.json")
                    .read_text(encoding="utf-8"))
    return [m for m in wc["matches"] if "group" not in m]


@st.cache_resource(max_entries=2,
                   show_spinner="⚙️ Recalculando el modelo con el nuevo "
                                "resultado (replay histórico + torneo)…")
def build_engine(live_tok: str) -> LiveEngine:
    """Histórico + live + estado del torneo + corrección online. Se invalida
    cuando cambia cualquier archivo de data/live/ (live_tok)."""
    art = load_artifacts()
    matches = pd.read_parquet(ROOT / "data" / "interim" / "matches.parquet")
    sv_path = ROOT / "data" / "processed" / "squad_values.parquet"
    squad_values = pd.read_parquet(sv_path) if sv_path.exists() else None
    t0 = time.perf_counter()
    eng = LiveEngine(matches, art["clf"], art["pois_home"],
                     art["pois_away"], art["rho"], art["blend"], STORE,
                     xgb=art.get("xgb"), weights=art.get("weights"),
                     squad_values=squad_values)
    LOG.info("LiveEngine construido (token=%s) en %.1fs: %s históricos, "
             "%s en vivo", live_tok, time.perf_counter() - t0,
             len(matches), len(STORE.results()))
    return eng


@st.cache_data(show_spinner="🎲 Simulando el torneo completo (Monte Carlo "
                            "con incertidumbre de fuerza)…")
def run_simulation(live_tok: str, n_sims: int) -> tuple[pd.DataFrame, dict]:
    """Monte Carlo del torneo completo; se invalida al ingresar resultados."""
    eng = build_engine(live_tok)
    teams = load_teams()
    groups = teams.groupby("group")["name_canonical"].apply(list).to_dict()
    sim = TournamentSimulator(eng, load_fixtures(), STORE.results(),
                              load_ko_raw(), groups)
    t0 = time.perf_counter()
    df = sim.run(n_sims)
    LOG.info("Monte Carlo: %s sims en %.1fs (token=%s); favorito=%s",
             n_sims, time.perf_counter() - t0, live_tok,
             df.iloc[0]["team"] if len(df) else "?")
    return df, sim.slot_stats


@st.cache_data
def build_bracket_payload(live_tok: str, n_sims: int) -> dict:
    """Payload del bracket (spec §7) con propagación DETERMINISTA
    (invariante U4): los entrantes a R32 son los ocupantes modales del
    Monte Carlo, pero de ahí en adelante en cada llave avanza el equipo
    con P(avanza) > 50% en ESE cruce (o el ganador real si ya se ingresó).
    Las marginales de slot_stats NO componen entre rondas: solo se usan
    para los pct mostrados."""
    from mundial.predict.montecarlo import HOST_OF_COUNTRY, _ground_country

    _, slots = run_simulation(live_tok, n_sims)
    eng = build_engine(live_tok)
    fx = load_fixtures()

    def _flag_url(team: str | None, size: int = 40) -> str | None:
        iso = FLAG_ISO.get(team or "")
        return f"https://flagcdn.com/w{size}/{iso}.png" if iso else None

    # ganadores reales de KO ya ingresados (mismo criterio del simulador)
    lv = STORE.results()
    gkeys = set(zip(fx.home_team, fx.away_team))
    ko_real: dict[frozenset, str] = {}
    for r in lv.itertuples(index=False):
        if (r.home_team, r.away_team) in gkeys:
            continue
        w = getattr(r, "ko_winner", None)
        if not isinstance(w, str) or not w:
            hs, as_ = int(r.home_score), int(r.away_score)
            w = (r.home_team if hs > as_
                 else r.away_team if as_ > hs else None)
        if w:
            ko_real[frozenset((r.home_team, r.away_team))] = w

    ko = sorted(load_ko_raw(), key=lambda m: m.get("num", 999))
    ko = [m for m in ko if m["round"] != "Match for third place"]
    round_lbl = {"Round of 32": "Dieciseisavos", "Round of 16": "Octavos",
                 "Quarter-final": "Cuartos", "Semi-final": "Semifinal",
                 "Final": "Final"}

    det_win: dict[str, str] = {}          # "W74" -> equipo que avanza

    def _occupant(ref: str, modal: list) -> str | None:
        """Lado de la llave: ganador determinista de la llave previa, o el
        ocupante modal del Monte Carlo (entrantes desde grupos)."""
        return det_win.get(str(ref)) or (modal[0][0] if modal else None)

    per_round: dict[str, list] = {k: [] for k in round_lbl}
    champion = None
    for i, m in enumerate(ko):    # orden por num: cada ronda llega resuelta
        mk = str(m.get("num", f"x{i}"))
        ss = slots.get(mk, {"t1": [], "t2": [], "w": []})
        t1 = _occupant(m["team1"], ss["t1"])
        t2 = _occupant(m["team2"], ss["t2"])
        # U4-display: pct = P(avanzar en ESTE cruce) — mostrar la ocupación
        # marginal junto al ganador del head-to-head confunde (un equipo
        # puede ocupar menos el slot y aun así ser favorito del cruce)
        win, pwin, s1, s2 = None, None, None, None
        played = False
        if t1 and t2:
            win = ko_real.get(frozenset((t1, t2)))
            if win:
                pwin = 100
                played = True
            else:
                host = HOST_OF_COUNTRY[_ground_country(m.get("ground", ""))]
                date = pd.Timestamp(m["date"])
                if t2 == host:    # localía solo si un anfitrión juega en casa
                    p1 = 1 - eng.predict_match(date, t2, t1,
                                               False)["p_home_advances"]
                else:
                    p1 = eng.predict_match(date, t1, t2,
                                           t1 != host)["p_home_advances"]
                win = t1 if p1 >= 0.5 else t2
                pwin = round(100 * max(p1, 1 - p1))
                s1, s2 = round(100 * p1), round(100 * (1 - p1))
        num = m.get("num")
        if num is not None and win:
            det_win[f"W{num}"] = win
        srcs = [int(str(x)[1:]) if str(x).startswith("W") else None
                for x in (m["team1"], m["team2"])]
        per_round[m["round"]].append({
            "num": num if num is not None else "final",
            "src1": srcs[0], "src2": srcs[1],
            "t1": {"team": t1, "flag": _flag_url(t1), "pct": s1} if t1 else None,
            "t2": {"team": t2, "flag": _flag_url(t2), "pct": s2} if t2 else None,
            "win": win, "pwin": pwin, "played": played,
            "cands1": ss["t1"][:3], "cands2": ss["t2"][:3],
            "date": pd.Timestamp(m["date"]).strftime("%d %b").upper(),
            "ground": m.get("ground", "")})
        if m["round"] == "Final" and win:
            # para el campeón sí es útil la marginal: P(campeón) del sim
            fshare = dict(ss["w"]).get(win)
            champion = {"team": win, "flag": _flag_url(win),
                        "pct": round(100 * fshare, 1) if fshare else None}

    LOG.info("bracket determinista: %s llaves, %s KO reales, campeón=%s",
             len(det_win), len(ko_real),
             champion["team"] if champion else "?")
    return {"rounds": [{"key": rnd.replace(" ", "_"), "label": lbl,
                        "matches": per_round[rnd]}
                       for rnd, lbl in round_lbl.items()],
            "champion": champion}


@st.cache_data
def build_final_payload(live_tok: str) -> dict | None:
    """Payload del dashboard de definición (spec §10).

    F2: los equipos se DERIVAN de las filas stage=="SF" de live_results
    (ganador → Final, perdedor → 3.er puesto; SF1 ocupa el slot local).
    F1: probabilidades exactas de los 2 partidos restantes (sin Monte
    Carlo). F3: Bota de Oro por Poisson truncadas. F4: el camino usa las
    predicciones pre-partido de live_audit, nunca el estado actual.
    Devuelve None si aún no hay 2 semifinales cargadas (la UI degrada)."""
    from mundial.predict.finalists import (
        golden_boot_race, podium_probs, podium_scenarios,
    )

    eng = build_engine(live_tok)
    lv = STORE.results()
    if not len(lv) or "stage" not in lv.columns:
        return None
    sf = lv[lv.stage.astype(str).str.upper() == "SF"].sort_values(
        "date", kind="stable")
    if len(sf) < 2:
        return None

    def _winner(r) -> str:
        w = getattr(r, "ko_winner", None)
        if isinstance(w, str) and w:
            return w
        return (r.home_team if int(r.home_score) > int(r.away_score)
                else r.away_team)

    sf_rows = list(sf.itertuples(index=False))[:2]
    finalists = tuple(_winner(r) for r in sf_rows)
    third = tuple(r.away_team if _winner(r) == r.home_team else r.home_team
                  for r in sf_rows)

    ko = load_ko_raw()
    m_final = next(m for m in ko if m["round"] == "Final")
    m_third = next((m for m in ko if m["round"] == "Match for third place"),
                   m_final)
    d_final = pd.Timestamp(m_final["date"])
    d_third = pd.Timestamp(m_third["date"])
    sf_last = max(pd.Timestamp(r.date) for r in sf_rows)

    def _played(a: str, b: str):
        """Fila live del cruce {a,b} posterior a las semis (Final o 3.er
        puesto ya ingresados), sin depender de la etiqueta de stage."""
        for r in lv.itertuples(index=False):
            if ({r.home_team, r.away_team} == {a, b}
                    and pd.Timestamp(r.date) > sf_last):
                return r
        return None

    pf = eng.predict_match(d_final, finalists[0], finalists[1], True)
    pt = eng.predict_match(d_third, third[0], third[1], True)
    dist_f = eng.match_distribution(d_final, finalists[0], finalists[1], True)
    dist_t = eng.match_distribution(d_third, third[0], third[1], True)

    def _match_view(p: dict, dist: dict, teams: tuple[str, str],
                    meta: dict, real) -> dict:
        mat = dist["matrix"]
        n = mat.shape[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        return {
            "teams": teams, "p": p,
            "date": pd.Timestamp(meta["date"]).strftime("%d %b").upper(),
            "ground": meta.get("ground", ""),
            "matrix": mat[:6, :6].tolist(),
            "p_btts": float(mat[1:, 1:].sum()),
            "p_over25": float(mat[(i + j) >= 3].sum()),
            "real": None if real is None else {
                "score": f"{int(real.home_score)}–{int(real.away_score)}",
                "winner": _winner(real)},
        }

    fp = _played(*finalists)
    tp = _played(*third)
    # probabilidades de puesto: exactas si falta el partido, 0/1 si ya se jugó
    p_final_home = (pf["p_home_advances"] if fp is None
                    else float(_winner(fp) == finalists[0]))
    p_third_home = (pt["p_home_advances"] if tp is None
                    else float(_winner(tp) == third[0]))
    podium = podium_probs(finalists, p_final_home, third, p_third_home)
    scenarios = podium_scenarios(finalists, p_final_home, third, p_third_home)

    # ---- Bota de Oro (F3): share del jugador × λ del partido que le queda
    pl = STORE.players()
    tally = (pl[pl.event.isin(["goal", "penalty"])]
             .groupby(["player", "team"]).size().reset_index(name="goals"))
    team_goals = tally.groupby("team")["goals"].sum().to_dict()
    lam_left = {
        finalists[0]: 0.0 if fp is not None else dist_f["lambda_home"],
        finalists[1]: 0.0 if fp is not None else dist_f["lambda_away"],
        third[0]: 0.0 if tp is not None else dist_t["lambda_home"],
        third[1]: 0.0 if tp is not None else dist_t["lambda_away"],
    }
    alive = tally[tally.team.isin(lam_left) & (tally.goals >= 4)]
    out = tally[~tally.team.isin(lam_left) & (tally.goals >= 6)]
    contenders = [
        {"player": r.player, "team": r.team, "goals": int(r.goals),
         "lam": lam_left.get(r.team, 0.0)
         * float(r.goals) / max(team_goals.get(r.team, 1), 1)}
        for r in pd.concat([alive, out]).itertuples(index=False)
    ]
    race = golden_boot_race(contenders) if contenders else pd.DataFrame()
    if len(race):
        assists = pl[pl.event == "assist"].groupby("player").size()
        race["assists"] = race.player.map(assists).fillna(0).astype(int)

    # ---- el camino de los 4 (F4): live_audit + stage/ko_winner del store
    def _kow(r) -> str:
        """ko_winner saneado: la celda vacía llega como NaN (truthy!)."""
        w = getattr(r, "ko_winner", None)
        return w if isinstance(w, str) else ""

    meta_by_pair = {(r.home_team, r.away_team): (str(r.stage), _kow(r))
                    for r in lv.itertuples(index=False)}
    stage_lbl = {"GROUP": "GRUPO", "R32": "16.OS", "R16": "8.OS",
                 "QF": "4.TOS", "SF": "SEMIS", "F": "FINAL"}
    journeys: dict[str, dict] = {}
    for team in (*finalists, *third):
        steps, hits, n_ko_extra = [], 0, 0
        surprise = None          # triunfo con menor prob pre-partido
        for m in eng.live_audit:
            if team not in (m["home_team"], m["away_team"]):
                continue
            is_home = m["home_team"] == team
            rival = m["away_team"] if is_home else m["home_team"]
            gf, gc = ((m["home_score"], m["away_score"]) if is_home
                      else (m["away_score"], m["home_score"]))
            stage, kow = meta_by_pair.get((m["home_team"], m["away_team"]),
                                          ("group", ""))
            p_team = m["p_home"] if is_home else m["p_away"]
            probs = {"H": m["p_home"], "D": m["p_draw"], "A": m["p_away"]}
            real = "H" if gf > gc else ("A" if gc > gf else "D")
            if not is_home:
                real = {"H": "A", "A": "H", "D": "D"}[real]
            hit = max(probs, key=probs.get) == real
            hits += hit
            won = gf > gc or (gf == gc and kow == team)
            if gf == gc and kow:
                n_ko_extra += 1
            if won and (surprise is None or p_team < surprise[1]):
                surprise = (rival, p_team, f"{gf}–{gc}")
            steps.append({
                "stage": stage_lbl.get(stage.upper(), stage.upper()),
                "rival": rival, "gf": int(gf), "gc": int(gc),
                "res": "G" if won else ("E" if gf == gc else "P"),
                "pens": bool(gf == gc and kow), "p_pre": p_team, "hit": hit,
            })
        gf_t = sum(s["gf"] for s in steps)
        gc_t = sum(s["gc"] for s in steps)
        date_next = d_final if team in finalists else d_third
        exp = eng.state.explain(team, date_next)
        journeys[team] = {
            "steps": steps, "hits": hits, "gf": gf_t, "gc": gc_t,
            "won": sum(s["res"] == "G" for s in steps),
            "draw": sum(s["res"] == "E" for s in steps),
            "lost": sum(s["res"] == "P" for s in steps),
            "ko_extra": n_ko_extra, "surprise": surprise,
            "elo": eng.elo_for(team, date_next), "momentum": exp["momentum"],
            "adjust": exp["total"],
        }

    return {
        "finalists": finalists, "third": third,
        "final": _match_view(pf, dist_f, finalists, m_final, fp),
        "third_match": _match_view(pt, dist_t, third, m_third, tp),
        "podium": podium, "scenarios": scenarios, "race": race,
        "journeys": journeys,
        "n_audit": len(eng.live_audit),
        "hits_audit": sum(
            max({"H": m["p_home"], "D": m["p_draw"], "A": m["p_away"]},
                key=lambda k: {"H": m["p_home"], "D": m["p_draw"],
                               "A": m["p_away"]}[k])
            == ("H" if m["home_score"] > m["away_score"]
                else "A" if m["away_score"] > m["home_score"] else "D")
            for m in eng.live_audit),
    }


@st.cache_data
def backtest_last(team: str, n: int = 5) -> pd.DataFrame:
    """Backtesting honesto (spec §9): reconstruye la predicción PRE-partido
    con las features de la capa Gold (anti-leakage verificado en el
    pipeline) y los artefactos entrenados — el MISMO ensemble de
    producción. Nunca usa el Elo actual del engine (sería leakage)."""
    from mundial.models.baseline import FEATURES
    from mundial.models.poisson import (
        POISSON_FEATURES, outcome_probs, score_matrix,
    )
    art = load_artifacts()
    f = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    f = (f[((f.home_team == team) | (f.away_team == team))
           & f.home_score.notna()]
         .dropna(subset=FEATURES)
         .sort_values("date").tail(n))
    rows = []
    if not f.empty:
        p_clf = art["clf"].predict_proba(f[FEATURES])    # columnas A, D, H
        lh = art["pois_home"].predict(f[POISSON_FEATURES])
        la = art["pois_away"].predict(f[POISSON_FEATURES])
        for k, (_, r) in enumerate(f.iterrows()):
            pp = outcome_probs(score_matrix(float(lh[k]), float(la[k]),
                                            art["rho"]))
            p = (art["blend"] * p_clf[k]
                 + (1 - art["blend"]) * np.array([pp["A"], pp["D"], pp["H"]]))
            pa, pd_, ph = float(p[0]), float(p[1]), float(p[2])
            rows.append(_audit_row(team, r.date, r.home_team, r.away_team,
                                   int(r.home_score), int(r.away_score),
                                   ph, pd_, pa))
    # partidos del torneo ingresados en vivo: la predicción honesta
    # PRE-partido que el LiveEngine registró durante el replay
    for m in engine.live_audit:
        if team in (m["home_team"], m["away_team"]):
            rows.append(_audit_row(team, m["date"], m["home_team"],
                                   m["away_team"], m["home_score"],
                                   m["away_score"], m["p_home"],
                                   m["p_draw"], m["p_away"], live=True))
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("date")
            .tail(n).sort_values("date", ascending=False))


@st.cache_data
def calibration_table(n_bins: int = 8) -> pd.DataFrame:
    """Curva de confiabilidad en test temporal >= 2022: cuando el ensemble
    dice X% para su 1X2 más probable, ¿acierta X% de las veces?"""
    from mundial.models.baseline import FEATURES
    from mundial.models.poisson import POISSON_FEATURES, predict_proba_1x2
    art = load_artifacts()
    f = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    f = f[(f.year >= 2022) & f.home_score.notna()].dropna(subset=FEATURES)
    p = (art["blend"] * art["clf"].predict_proba(f[FEATURES])
         + (1 - art["blend"]) * predict_proba_1x2(
             art["pois_home"], art["pois_away"], f[POISSON_FEATURES],
             art["rho"]))
    conf = p.max(axis=1)
    pred = np.array(["A", "D", "H"])[p.argmax(axis=1)]
    real = np.where(f.home_score > f.away_score, "H",
                    np.where(f.home_score < f.away_score, "A", "D"))
    df = pd.DataFrame({"conf": conf, "ok": pred == real})
    df["bin"] = pd.cut(df.conf, bins=np.linspace(0.33, 1.0, n_bins + 1))
    g = df.groupby("bin", observed=True).agg(
        confianza=("conf", "mean"), acierto=("ok", "mean"),
        n=("ok", "size"))
    return g[g.n >= 30].reset_index(drop=True)


def _audit_row(team, date, home, away, hs, as_, ph, pd_, pa,
               live: bool = False) -> dict:
    pred = max((("H", ph), ("D", pd_), ("A", pa)), key=lambda x: x[1])[0]
    real = "H" if hs > as_ else ("A" if as_ > hs else "D")
    es_local = home == team
    return {"date": pd.Timestamp(date),
            "rival": away if es_local else home,
            "es_local": es_local,
            "p_win": ph if es_local else pa, "p_draw": pd_,
            "score": f"{hs} – {as_}",
            "pred": pred, "real": real, "ok": pred == real, "live": live}


@st.cache_data
def load_rosters() -> dict[str, list[str]]:
    """Plantillas normalizadas Gold (spec §2.4) para los dropdowns."""
    path = ROOT / "data" / "processed" / "rosters_2026.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return df.groupby("team")["player"].apply(list).to_dict()


def flag_img(team: str, size: int = 40) -> str:
    """<img> de la bandera (flagcdn). Solo para HTML, no para labels."""
    iso = FLAG_ISO.get(team)
    if not iso:
        return ""
    # flagcdn solo soporta w40, w80, w160, w320
    cdn_w = min([40, 80, 160, 320], key=lambda w: abs(w - size))
    return (f'<img src="https://flagcdn.com/w{cdn_w}/{iso}.png" '
            f'width="{size}" alt="{esc(team)}">')


def tbl(df: pd.DataFrame, flags: set[str] | None = None,
        bars: set[str] | None = None, height: int | None = None) -> str:
    """Tabla HTML tematizada (reemplaza st.dataframe para respetar el tema)."""
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if flags and c in flags:
                tds.append(f'<td class="t-team">{flag_img(str(v), 20)} '
                           f'{esc(str(v))}</td>')
            elif bars and c in bars:
                x = max(0.0, min(1.0, float(v)))
                tds.append(f'<td class="t-bar"><span class="pbar">'
                           f'<div style="width:{100 * x:.0f}%"></div></span>'
                           f'{100 * x:.1f}%</td>')
            else:
                if isinstance(v, float):
                    v = f"{v:g}"
                tds.append(f"<td>{esc(str(v))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    style = f' style="max-height:{height}px"' if height else ""
    return (f'<div class="tblwrap"{style}><table class="tbl">'
            f'<thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def match_card(r, p, adj: tuple[float, float] | None = None,
               xai_key: str = "") -> str:
    """HTML de una card de partido con diseño tipográfico premium.
    adj: ajuste Elo en vivo (local, visitante) para el chip XAI.
    xai_key: identificador para vincular al modal XAI."""
    fecha = pd.Timestamp(r.date).strftime("%a %d %b").upper()
    sede = f"{r.city}" + ("" if r.neutral else " · CASA")
    grupo = f"GRUPO {r.group} · " if r.group else ""
    ph, pd_, pa = p["p_home"], p["p_draw"], p["p_away"]
    plabel, pcss = display_pred(ph, pd_, pa, r.home_team, r.away_team)
    adj_html = ""
    if adj and (abs(adj[0]) >= 0.5 or abs(adj[1]) >= 0.5):
        adj_html = (f'<span class="mc-adj">Δ {adj[0]:+.0f} / {adj[1]:+.0f}'
                    f' ELO</span>')
    return (
        f'<div class="glass match-card">'
        f'<div class="mc-meta">{fecha} · {grupo}{sede}{adj_html}</div>'
        f'<div class="mc-row">'
        f'<div class="mc-team"><span class="flag">{flag_img(r.home_team)}</span>'
        f'{esc(r.home_team)}</div>'
        f'<div class="mc-team"><span class="flag">{flag_img(r.away_team)}</span>'
        f'{esc(r.away_team)}</div>'
        f'</div>'
        f'<div class="mc-probs-wrap">'
        f'<div class="mc-prob-block h"><span class="val">{100 * ph:.0f}%</span>'
        f'<span class="lbl">Local</span></div>'
        f'<div class="mc-prob-block d"><span class="val">{100 * pd_:.0f}%</span>'
        f'<span class="lbl">Empate</span></div>'
        f'<div class="mc-prob-block a"><span class="val">{100 * pa:.0f}%</span>'
        f'<span class="lbl">Visit.</span></div>'
        f'</div>'
        f'<div class="mc-pred {pcss}">🔮 {esc(plabel)}</div>'
        f'</div>'
    )


# ------------------------------------------------ formulario de detalle
def dynamic_rows(title: str, key: str, fields: list[tuple]) -> list[dict]:
    """Editor de filas con widgets nativos (DOM), tematizable en light y
    dark — reemplaza a st.data_editor, cuyo canvas no respeta el tema.
    fields: [(nombre, 'select'|'text'|'int', opciones)]."""
    rows: list[dict] = st.session_state.setdefault(key, [])
    st.markdown(f"**{title}**")
    widths = [3] * len(fields) + [1]
    for i, row in enumerate(rows):
        cols = st.columns(widths, vertical_alignment="center")
        for kk, (name, _, _) in enumerate(fields):
            cols[kk].markdown(f'<div class="rowchip">{esc(str(row[name]))}'
                              '</div>', unsafe_allow_html=True)
        if cols[-1].button("✕", key=f"{key}_del{i}"):
            rows.pop(i)
            st.rerun()
    nonce = st.session_state.get(f"{key}_nonce", 0)
    vals: dict = {}
    equipo_key = f"{key}_{nonce}_Equipo"
    # 2 campos por fila: el formulario vive en media página en desktop y
    # 4-5 columnas dejaban los selectboxes ilegibles (en móvil sí apilaba)
    for start in range(0, len(fields), 2):
        chunk = fields[start:start + 2]
        cols = st.columns(len(chunk), vertical_alignment="bottom")
        for kk, (name, kind, opts) in enumerate(chunk):
            wkey = f"{key}_{nonce}_{name}"
            if kind == "select":
                vals[name] = cols[kk].selectbox(name, opts, key=wkey)
            elif kind == "int":
                vals[name] = cols[kk].number_input(name, 1, 130, 1, key=wkey)
            elif kind == "player":
                # dropdown anti-typos: plantilla del equipo de esta fila
                team_sel = st.session_state.get(equipo_key,
                                                opts[0] if opts else "")
                roster = load_rosters().get(team_sel, [])
                choices = roster + [OTRO]
                pick = cols[kk].selectbox(name, choices, key=wkey) \
                    if roster else OTRO
                if pick == OTRO:
                    pick = cols[kk].text_input(f"{name} (otro)",
                                               key=f"{wkey}_free",
                                               placeholder="Nombre")
                vals[name] = pick
            else:
                vals[name] = cols[kk].text_input(name, key=wkey,
                                                 placeholder="Nombre")
    if st.button("＋ Agregar", key=f"{key}_add{nonce}",
                 use_container_width=True,
                 help="Agregar fila"):
        if all(str(vals[n]).strip() and vals[n] != OTRO
               for n, kind, _ in fields if kind in ("text", "player")):
            rows.append(vals)
            st.session_state[f"{key}_nonce"] = nonce + 1
            st.rerun()
        else:
            st.warning("Completa el nombre del jugador antes de agregar.")
    return rows


def detail_block(prefix: str, home: str, away: str) -> dict:
    """Editores de contexto del partido (todo opcional). Devuelve dicts
    listos para LiveStore.add_match."""
    teams = [home, away]
    out = {"xg_home": pd.NA, "xg_away": pd.NA, "weather": pd.NA,
           "formation_home": pd.NA, "formation_away": pd.NA,
           "players": [], "cards": [], "injuries": []}
    nonce = st.session_state.get("form_nonce", 0)
    k = f"{prefix}{nonce}"
    with st.expander("Contexto del partido — xG, goleadores, tarjetas, "
                     "lesiones, clima, formación (opcional)"):
        # transparencia (spec §4): qué usa de verdad el modelo vs qué se
        # guarda solo como historial
        st.caption("xG → momentum y corrección de ritmo de goles · "
                   "tarjetas → suspensiones FIFA · lesiones → ajuste Elo. "
                   "Clima y formación se guardan **solo como metadatos** "
                   "(no afectan la predicción).")
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.checkbox("Registrar xG", key=f"{k}_usexg"):
            out["xg_home"] = c1.number_input(f"xG {home}", 0.0, 15.0, 1.0,
                                             0.1, key=f"{k}_xgh")
            out["xg_away"] = c2.number_input(f"xG {away}", 0.0, 15.0, 1.0,
                                             0.1, key=f"{k}_xga")
        w = c3.selectbox("Clima · solo metadatos", WEATHER_OPTS,
                         key=f"{k}_wx")
        if w != "Sin dato":
            out["weather"] = w
        c1, c2 = st.columns(2)
        fh = c1.text_input(f"Formación {home} · solo metadatos",
                           placeholder="4-3-3", key=f"{k}_fh")
        fa = c2.text_input(f"Formación {away} · solo metadatos",
                           placeholder="4-4-2", key=f"{k}_fa")
        out["formation_home"] = fh or pd.NA
        out["formation_away"] = fa or pd.NA

        ev = dynamic_rows("Goles y asistencias", f"rows_ev_{prefix}", [
            ("Equipo", "select", teams), ("Jugador", "player", teams),
            ("Minuto", "int", None), ("Tipo", "select", list(EVENT_LABELS))])
        cards = dynamic_rows("Tarjetas", f"rows_cd_{prefix}", [
            ("Equipo", "select", teams), ("Jugador", "player", teams),
            ("Tarjeta", "select", list(CARD_LABELS)),
            ("Minuto", "int", None)])
        inj = dynamic_rows("Lesiones de jugadores clave", f"rows_in_{prefix}",
                           [("Equipo", "select", teams),
                            ("Jugador", "player", teams),
                            ("Gravedad", "select", list(INJURY_LABELS))])

    out["players"] = [{"team": r["Equipo"], "player": r["Jugador"],
                       "event": EVENT_LABELS[r["Tipo"]],
                       "minute": int(r["Minuto"])} for r in ev]
    out["cards"] = [{"team": r["Equipo"], "player": r["Jugador"],
                     "card": CARD_LABELS[r["Tarjeta"]],
                     "minute": int(r["Minuto"])} for r in cards]
    out["injuries"] = [{"team": r["Equipo"], "player": r["Jugador"],
                        "severity": INJURY_LABELS[r["Gravedad"]]}
                       for r in inj]
    return out


def save_match(date, home, away, gh, ga, neutral, stage, details,
               ko_winner=None, prefix: str = "g") -> None:
    STORE.add_match(
        {"date": pd.Timestamp(date), "home_team": home, "away_team": away,
         "home_score": int(gh), "away_score": int(ga), "neutral": bool(neutral),
         "stage": stage, "ko_winner": ko_winner or pd.NA,
         "xg_home": details["xg_home"], "xg_away": details["xg_away"],
         "weather": details["weather"],
         "formation_home": details["formation_home"],
         "formation_away": details["formation_away"]},
        players=details["players"], cards=details["cards"],
        injuries=details["injuries"])
    LOG.info("resultado guardado: %s %s-%s %s (%s)", home, int(gh), int(ga),
             away, stage)
    st.session_state["form_nonce"] = st.session_state.get("form_nonce", 0) + 1
    for kind in ("ev", "cd", "in"):       # limpia los editores de filas
        st.session_state.pop(f"rows_{kind}_{prefix}", None)
    flash(f"✅ {home} {int(gh)} – {int(ga)} {away} guardado. "
          "Predicciones recalculadas.")


def flash(msg: str) -> None:
    """Encola un mensaje que SOBREVIVE al st.rerun() (un st.success seguido
    de rerun se borra antes de que el usuario lo vea). Añade el estado del
    sync a GitHub para avisar si el dato quedó solo local."""
    if STORE.last_sync_ok is False:
        msg += (" ⚠️ No se pudo publicar en GitHub: el dato quedó solo en "
                "esta sesión (revisa el token en Secrets).")
    st.session_state["_flash"] = msg


def show_flash() -> None:
    msg = st.session_state.pop("_flash", None)
    if msg:
        (st.warning if "⚠️" in msg else st.success)(msg)
        st.toast(msg)


# ------------------------------------------------ XAI dialog
@st.dialog("Explicabilidad del pronóstico", width="large")
def xai_dialog(home: str, away: str, date, p: dict, adj_h: float, adj_a: float,
               ls: dict):
    st.markdown(
        f'<div class="xai-team">{flag_img(home, 22)} {esc(home)} '
        f'<span style="color:var(--muted);font-weight:400">vs</span> '
        f'{flag_img(away, 22)} {esc(away)}</div>',
        unsafe_allow_html=True)
    st.markdown(f'<div class="mc-meta">{pd.Timestamp(date).strftime("%a %d %b %Y")}'
                f'</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {home}", f"{100 * p['p_home']:.0f}%")
    c2.metric("Empate", f"{100 * p['p_draw']:.0f}%")
    c3.metric(f"Gana {away}", f"{100 * p['p_away']:.0f}%")

    st.markdown('<div class="xai-divider"></div>', unsafe_allow_html=True)

    e_home = engine.state.explain(home, pd.Timestamp(date))
    e_away = engine.state.explain(away, pd.Timestamp(date))

    col1, col2 = st.columns(2)
    for col, team, expl in [(col1, home, e_home), (col2, away, e_away)]:
        with col:
            st.markdown(f'<div class="xai-team" style="font-size:.95rem">'
                        f'{flag_img(team, 18)} {esc(team)}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="xai-stat"><span class="label">Elo base</span>'
                f'<span class="value">{engine.elo.get(team, 1500):.0f}</span></div>'
                f'<div class="xai-stat pos"><span class="label">Momentum</span>'
                f'<span class="value">+{expl["momentum"]:.1f}</span></div>'
                f'<div class="xai-stat neg"><span class="label">Sanciones/lesiones</span>'
                f'<span class="value">{expl["penalty"]:.1f}</span></div>'
                f'<div class="xai-divider"></div>'
                f'<div class="xai-stat"><span class="label">Ajuste total</span>'
                f'<span class="value">{expl["total"]:+.1f} Elo</span></div>'
                f'<div class="xai-stat"><span class="label">Elo efectivo</span>'
                f'<span class="value">{engine.elo_for(team, pd.Timestamp(date)):.0f}'
                f'</span></div>',
                unsafe_allow_html=True)

            if expl["items"]:
                st.markdown('<div class="mc-meta" style="margin-top:8px">'
                            'Detalle</div>', unsafe_allow_html=True)
                for lbl, pts in expl["items"]:
                    st.markdown(
                        f'<div class="xai-stat"><span class="label">{esc(lbl)}</span>'
                        f'<span class="value">{pts:+.0f}</span></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="xai-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-meta">Correcciones globales del torneo</div>',
                unsafe_allow_html=True)
    if ls.get("n", 0):
        st.markdown(
            f'<div class="xai-stat"><span class="label">Ritmo de goles (γ)</span>'
            f'<span class="value">×{ls["gamma"]:.3f}</span></div>'
            f'<div class="xai-stat"><span class="label">Frecuencia empates</span>'
            f'<span class="value">×{ls["draw_mult"]:.3f}</span></div>'
            f'<div class="xai-stat"><span class="label">Factor altitud</span>'
            f'<span class="value">×{ls["alt_mult"]:.3f}</span></div>'
            f'<div class="xai-stat"><span class="label">Partidos observados</span>'
            f'<span class="value">{ls["n"]}</span></div>',
            unsafe_allow_html=True)
    else:
        st.caption("Aún no hay correcciones (el torneo no comenzó).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="mc-meta">Goles esperados (Poisson)</div>'
                    f'<div class="xai-stat"><span class="label">{esc(home)}</span>'
                    f'<span class="value">{p["lambda_home"]:.2f}</span></div>'
                    f'<div class="xai-stat"><span class="label">{esc(away)}</span>'
                    f'<span class="value">{p["lambda_away"]:.2f}</span></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="mc-meta">Marcadores más probables</div>',
                    unsafe_allow_html=True)
        for s, pr in p["scorelines"]:
            st.markdown(f'<span class="pill">{s.replace("-", " – ")} · '
                        f'{100 * pr:.0f}%</span>', unsafe_allow_html=True)

    # ---- sección pedagógica: qué significa todo esto (para no expertos)
    with st.expander("📚 ¿Qué es el rating Elo y cómo afecta esta predicción?"):
        # sin div wrapper: un <div> abierto en un markdown y cerrado en otro
        # NO envuelve nada (Streamlit lo autocierra) y dejaba una caja vacía
        st.markdown(
            "**Elo** es un sistema de puntaje (nacido en el ajedrez) donde "
            "cada selección tiene un número que sube al ganar y baja al "
            "perder. Vencer a un rival fuerte da muchos puntos; vencer a "
            "uno débil, pocos. Todas las selecciones parten de 1500.")
        st.markdown("**1 · Resultado esperado** — antes del partido, el "
                    "modelo convierte la diferencia de Elo en probabilidad:")
        st.latex(r"E_{local} = \frac{1}{1 + 10^{(Elo_{visit} - Elo_{local}"
                 r" - ventaja_{casa}) / 400}}")
        st.caption("Si ambos tienen el mismo Elo, E = 50%. Cada ~100 puntos "
                   "de ventaja suben la expectativa a ~64%; +200 ≈ 76%. La "
                   "ventaja de casa vale 65 puntos extra.")
        st.markdown("**2 · Actualización tras el partido** — el Elo se "
                    "corrige según cuánto sorprendió el resultado:")
        st.latex(r"Elo' = Elo + K \cdot g(\text{margen}) \cdot "
                 r"(S - E)\quad K=30")
        st.caption("S = 1 si ganó, 0.5 empate, 0 si perdió. g(margen) "
                   "amplifica goleadas (2 goles ×1.5, 3+ aún más). Ganar "
                   "siendo favorito casi no mueve el Elo; ganar de visita "
                   "contra un grande lo dispara.")
        st.markdown("**3 · Cómo entra en ESTA predicción** — el Elo "
                    "efectivo de cada equipo (base + momentum del torneo − "
                    "sanciones/lesiones, el desglose de arriba) alimenta "
                    "las features del clasificador y del modelo Poisson de "
                    "goles. Una diferencia de Elo mayor desplaza las "
                    "probabilidades y los goles esperados (λ) hacia el "
                    "favorito; las correcciones online del torneo ajustan "
                    "el ritmo de goles y los empates al final.")

    if st.button("Cerrar", type="primary"):
        st.rerun()


# ----------------------------------------------------------------- app
art = load_artifacts()
fixtures = load_fixtures()
live = STORE.results()
engine = build_engine(STORE.token())

hcol, acol, tcol = st.columns([4.4, 1.1, .9], vertical_alignment="center")
hcol.markdown('<h1 class="hero">Oráculo personal de Nicolás — '
              '<span class="grad">Mundial 2026</span></h1>',
              unsafe_allow_html=True)
with acol:
    auth.login_entry()       # spec §8: login modal, sin sidebar
tcol.toggle("Modo claro", key="light_mode")

ls = engine.live_summary()
factores = ""
if ls["n"]:
    factores = (f' · online: goles ×{ls["gamma"]:.3f}, '
                f'empates ×{ls["draw_mult"]:.3f}, altitud ×{ls["alt_mult"]:.3f}')
st.markdown(f'<p class="hero-sub">Predicciones en vivo · ensemble '
            f'Logistic + Poisson Dixon-Coles · {art["n_train"]:,} partidos '
            f'hasta {art["trained_until"]} · resultados ingresados: '
            f'{len(live)}{factores}</p>', unsafe_allow_html=True)

played_keys = set(zip(live.home_team, live.away_team)) if len(live) else set()
pending = fixtures[~fixtures.apply(
    lambda r: (r.home_team, r.away_team) in played_keys, axis=1)]

# ---- RBAC (spec §8): viewers no construyen el tab de ingesta
IS_ADMIN = auth.is_admin()
_labels = ["Final", "Próximos partidos", "Aciertos", "Mercado", "Líderes",
           "Cuadrangular", "Eliminatorias", "Camino al título", "Tablas",
           "Auditoría"]
if IS_ADMIN:
    _labels.insert(1, "Ingresar resultado")
_tabs = dict(zip(_labels, st.tabs(_labels)))
tab_final = _tabs["Final"]
tab_pred = _tabs["Próximos partidos"]
tab_hits = _tabs["Aciertos"]
tab_market = _tabs["Mercado"]
tab_leaders = _tabs["Líderes"]
tab_bracket = _tabs["Cuadrangular"]
tab_ko = _tabs["Eliminatorias"]
tab_champ = _tabs["Camino al título"]
tab_tablas = _tabs["Tablas"]
tab_audit = _tabs["Auditoría"]
tab_result = _tabs.get("Ingresar resultado")

# ------------------------------------------------ TAB 1: predicciones
with tab_pred:
    if pending.empty:
        # Fase de grupos completa: mostramos los cruces de eliminatorias ya
        # definidos (clasificados reales + ocupantes modales del Monte Carlo)
        # con su pronóstico. A medida que se ingresan resultados de KO, la
        # siguiente ronda se va resolviendo y aparece aquí (invariante U4).
        st.success("Fase de grupos completa. Pronósticos de eliminatorias "
                   "según se van definiendo los cruces:")
        _nsims = st.session_state.get("nsims_bracket", 5000)
        with st.spinner("Resolviendo cruces y calculando pronósticos…"):
            _bp = build_bracket_payload(STORE.token(), _nsims)
        _shown = 0
        for _rnd in _bp["rounds"]:
            _ms = [m for m in _rnd["matches"]
                   if m["t1"] and m["t2"] and not m["played"]]
            if not _ms:
                continue
            st.markdown(f"#### {_rnd['label']}")
            for _i in range(0, len(_ms), 3):
                _cols = st.columns(3)
                for _col, _m in zip(_cols, _ms[_i:_i + 3]):
                    _t1, _t2 = _m["t1"], _m["t2"]
                    _win, _pwin = _m["win"], _m["pwin"]
                    _s1 = _t1["pct"] if _t1["pct"] is not None else 50
                    _s2 = _t2["pct"] if _t2["pct"] is not None else 50
                    def _side(_t, _pct, _is_win):
                        _fl = (f'<img src="{_t["flag"]}" style="width:22px;'
                               f'border-radius:3px;vertical-align:middle">'
                               if _t["flag"] else "")
                        _cls = "color:#ff5470;font-weight:700" if _is_win else ""
                        return (f'<div style="display:flex;justify-content:'
                                f'space-between;{_cls}"><span>{_fl} '
                                f'{esc(_t["team"])}</span><span>{_pct}%</span></div>')
                    _col.markdown(
                        f'<div class="glass" style="padding:14px">'
                        f'<div class="mc-meta">{_m["date"]} · '
                        f'{esc(_m["ground"])}</div>'
                        f'{_side(_t1, _s1, _win == _t1["team"])}'
                        f'<div style="opacity:.5;text-align:center;'
                        f'font-size:.7rem;margin:2px 0">vs</div>'
                        f'{_side(_t2, _s2, _win == _t2["team"])}'
                        f'<div class="mc-meta" style="margin-top:8px">'
                        f'Avanza (proy.): <b>{esc(_win)}</b> · {_pwin}%</div>'
                        f'</div>', unsafe_allow_html=True)
            _shown += len(_ms)
        st.caption("Los equipos sin cruce real definido son los ocupantes más "
                   "probables del Monte Carlo; el % de cada lado es P(avanza) "
                   "en ese cruce. Usa la pestaña Cuadrangular para el cuadro "
                   "completo y Eliminatorias para simular un cruce manual.")
        if _shown == 0:
            st.info("Aún no hay cruces de eliminatorias con ambos equipos "
                    "definidos.")
    else:
        # jornadas derivadas del calendario REAL: la n-ésima vez que un
        # equipo juega es su Fecha n (los rivales van siempre parejos)
        _cnt: dict[str, int] = {}
        _md: dict[int, int] = {}
        for _ridx, _r in fixtures.sort_values("date").iterrows():
            _j = max(_cnt.get(_r.home_team, 0), _cnt.get(_r.away_team, 0)) + 1
            _cnt[_r.home_team] = _cnt[_r.away_team] = _j
            _md[_ridx] = _j
        _MES = {6: "jun", 7: "jul"}
        _rng: dict[int, tuple] = {}
        for _ridx, _j in _md.items():
            _d = fixtures.loc[_ridx, "date"]
            _lo, _hi = _rng.get(_j, (_d, _d))
            _rng[_j] = (min(_lo, _d), max(_hi, _d))
        _hz = {(f"Fecha {j} — del {lo.day} {_MES[lo.month]} al "
                f"{hi.day} {_MES[hi.month]}"): j
               for j, (lo, hi) in sorted(_rng.items())}
        _hz["Toda la fase de grupos"] = 0
        _pick = st.selectbox("📅 Jornada mundialista", list(_hz),
                             help="Qué jornada de la fase de grupos mostrar. "
                                  "Las fechas salen del calendario oficial.")
        _jor = _hz[_pick]
        sel = (pending if _jor == 0
               else pending[pending.index.map(_md) == _jor])
        if sel.empty:
            st.info("Todos los partidos de esta jornada ya tienen "
                    "resultado ingresado.")
        cards_data = []
        for idx in range(len(sel)):
            r = sel.iloc[idx]
            p = engine.predict_match(r.date, r.home_team, r.away_team,
                                     r.neutral, city=r.city)
            adj = (engine.state.adjustment(r.home_team, r.date),
                   engine.state.adjustment(r.away_team, r.date))
            card_html = match_card(r, p, adj)
            cards_data.append((r, p, adj, card_html, idx))

        for i in range(0, len(cards_data), 3):
            cols = st.columns(3)
            for col, (r, p, adj, html, idx) in zip(cols, cards_data[i:i + 3]):
                col.markdown(html, unsafe_allow_html=True)
                if col.button("📊 Explicar pronóstico",
                              key=f"xai_pred_{idx}",
                              use_container_width=True):
                    xai_dialog(r.home_team, r.away_team, r.date, p,
                               adj[0], adj[1], ls)

        st.caption("Las predicciones ya incluyen momentum, sanciones, lesiones "
                   "y corrección online. Haz clic en 'Explicar pronóstico' para "
                   "ver el desglose completo.")

# ------------------------------------------------ TAB 2: ingresar resultado
if IS_ADMIN:  # RBAC: el tab solo existe para admin (spec R1)
    with tab_result:
        show_flash()
        st.markdown(
            '<div class="form-card"><h4>📋 Registro oficial de partidos</h4>'
            '<p style="color:var(--muted);font-size:.82rem;margin:0">Ingresa '
            'resultados reales del torneo. El motor recalcula automáticamente '
            'predicciones, momentum, y probabilidades de clasificación.</p></div>',
            unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1], gap="large")

        # paneles con st.container(border=True): el recuadro envuelve de
        # verdad a los widgets (los <div> abiertos/cerrados en markdowns
        # separados no contienen nada — Streamlit los cierra solos)
        with col_a, st.container(border=True):
            st.markdown('<h4 class="fc-title">🏆 Fase de grupos</h4>',
                        unsafe_allow_html=True)
            if pending.empty:
                st.info("No hay fixtures pendientes de fase de grupos.")
            else:
                opts = {f"{r.date.date()} · Grupo {r.group} · {r.home_team} vs "
                        f"{r.away_team}": i for i, r in pending.iterrows()}
                pick = st.selectbox("Seleccionar partido", list(opts),
                                    label_visibility="collapsed")
                row = pending.loc[opts[pick]]
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:'
                    f'center;gap:16px;margin:12px 0">'
                    f'<span style="font-weight:700;font-size:1rem">'
                    f'{flag_img(row.home_team, 24)} {esc(row.home_team)}</span>'
                    f'<span style="color:var(--muted);font-weight:600">vs</span>'
                    f'<span style="font-weight:700;font-size:1rem">'
                    f'{flag_img(row.away_team, 24)} {esc(row.away_team)}</span>'
                    f'</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                gh = c1.number_input(f"Goles {esc(row.home_team)}", 0, 15, 0,
                                     key="gh", help="Goles que hizo el local")
                ga = c2.number_input(f"Goles {esc(row.away_team)}", 0, 15, 0,
                                     key="ga", help="Goles que hizo el visitante")
                details = detail_block("g", row.home_team, row.away_team)
                if st.button("💾 Guardar resultado de grupo",
                             type="primary", use_container_width=True):
                    save_match(row.date, row.home_team, row.away_team, gh, ga,
                               row.neutral, "group", details, prefix="g")
                    st.rerun()

        with col_b, st.container(border=True):
            st.markdown('<h4 class="fc-title">⚔️ Eliminatoria / manual</h4>',
                        unsafe_allow_html=True)
            teams48 = sorted(load_teams().name_canonical)
            c1, c2 = st.columns(2)
            mh = c1.selectbox("Local", teams48, key="mh")
            ma = c2.selectbox("Visitante", teams48, index=1, key="ma")
            c1, c2, c3 = st.columns([1, 1, 1])
            mgh = c1.number_input("Goles local", 0, 15, 0, key="mgh",
                                  help="Goles del equipo local")
            mga = c2.number_input("Goles visit.", 0, 15, 0, key="mga",
                                  help="Goles del equipo visitante")
            mdate = c3.date_input("Fecha", pd.Timestamp("2026-06-28"), key="mdate",
                                  help="Fecha del partido")
            winner = None
            if mgh == mga:
                wpick = st.selectbox("Ganador en prórroga/penales",
                                     ["No aplica", mh, ma], key="mwin",
                                     help="Solo para eliminatorias con empate")
                winner = None if wpick == "No aplica" else wpick
            mdetails = detail_block("m", mh, ma)
            if st.button("💾 Guardar partido manual",
                         type="primary", use_container_width=True):
                if mh == ma:
                    st.error("Elige dos selecciones distintas.")
                else:
                    save_match(pd.Timestamp(mdate), mh, ma, mgh, mga,
                               mh not in HOSTS, "ko", mdetails, ko_winner=winner,
                               prefix="m")
                    st.rerun()

        if len(live):
            hist = st.container(border=True)
            hist.markdown(f'<h4 class="fc-title">📜 Resultados ingresados '
                          f'({len(live)})</h4>', unsafe_allow_html=True)
            show = live[["date", "home_team", "home_score", "away_score",
                         "away_team", "xg_home", "xg_away", "weather",
                         "stage"]].copy()
            show["date"] = pd.to_datetime(show["date"]).dt.date
            # OJO: nombres únicos — columnas duplicadas hacen que r[c]
            # devuelva una Series y la celda imprima "Name: 0, dtype: ..."
            show = show.rename(columns={
                "date": "Fecha", "home_team": "Local", "home_score": "GL",
                "away_score": "GV", "away_team": "Visitante",
                "xg_home": "xG (L)", "xg_away": "xG (V)",
                "weather": "Clima", "stage": "Fase"})
            hist.markdown(tbl(show.fillna("—").sort_values("Fecha",
                                                           ascending=False),
                              flags={"Local", "Visitante"}, height=280),
                          unsafe_allow_html=True)
            # eliminar CUALQUIER resultado: borra en cascada (goleadores,
            # tarjetas, lesiones) y el partido sale del estado del modelo
            # en el siguiente replay — como si nunca se hubiera ingresado
            dcol1, dcol2 = hist.columns([3, 1], vertical_alignment="bottom")
            dopts = {
                f"{pd.Timestamp(r.date).date()} · {r.home_team} "
                f"{int(r.home_score)}-{int(r.away_score)} {r.away_team}":
                str(r.match_id)
                for r in live.sort_values("date").itertuples(index=False)}
            dpick = dcol1.selectbox("Resultado a eliminar", list(dopts),
                                    key="del_pick")
            if dcol2.button("🗑️ Eliminar", use_container_width=True):
                STORE.delete_match(dopts[dpick])
                flash(f"🗑️ Eliminado: {dpick}. El partido ya no afecta "
                      "al modelo.")
                st.rerun()

# ------------------------------------------------ TAB: aciertos
with tab_hits:
    st.subheader("Aciertos del modelo — pronóstico vs realidad")
    st.caption("Cada partido ingresado, con la predicción que el modelo "
               "hizo ANTES de conocer el resultado. Marcador: cuenta como "
               "acierto si el real está en el top-2 de marcadores exactos "
               "predichos.")
    audit = engine.live_audit
    if not audit:
        st.info("Ingresa resultados del torneo para ver la evaluación.")
    else:
        lbl1x2 = {"H": "Local", "D": "Empate", "A": "Visitante"}
        rows_h = []
        ok1, ok2 = 0, 0
        brier = 0.0
        draws_real = draws_pred = 0
        for m in audit:
            pmap = {"H": m["p_home"], "D": m["p_draw"], "A": m["p_away"]}
            probs = tuple(pmap.items())
            pred, pconf = max(probs, key=lambda x: x[1])
            hs, as_ = m["home_score"], m["away_score"]
            real = "H" if hs > as_ else ("A" if as_ > hs else "D")
            hit1 = pred == real
            # Brier multiclase: Σ(p−y)² sobre las 3 clases
            brier += sum((pmap[k] - (1.0 if k == real else 0.0)) ** 2
                         for k in pmap)
            draws_real += real == "D"
            draws_pred += pred == "D"
            tops = m.get("top_scores") or []
            hit2 = f"{hs}-{as_}" in {s for s, _ in tops}
            ok1 += hit1
            ok2 += hit2
            rows_h.append({
                "Fecha": pd.Timestamp(m["date"]).date(),
                "Local": m["home_team"],
                "Real": f"{hs} – {as_}",
                "Visitante": m["away_team"],
                "Pronóstico": f"{lbl1x2[pred]} ({100 * pconf:.0f}%)",
                "1X2": "✅" if hit1 else "❌",
                "Marcadores predichos": " · ".join(
                    f"{s.replace('-', '–')} {100 * pr:.0f}%"
                    for s, pr in tops) or "—",
                "Marcador": "✅" if hit2 else "❌",
            })
        n = len(rows_h)
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos evaluados", n)
        c2.metric("Acierto 1X2", f"{ok1}/{n}", f"{100 * ok1 / n:.0f}%")
        c3.metric("Acierto marcador (top-2)", f"{ok2}/{n}",
                  f"{100 * ok2 / n:.0f}%")
        d1, d2 = st.columns(2)
        d1.metric("Brier score", f"{brier / n:.3f}",
                  help="Calidad de las probabilidades (no del argmax). "
                       "Azar≈0.67, bueno<0.60, excelente<0.55. Fiable a "
                       "partir de ~20 partidos.")
        d2.metric("Empates reales vs predichos",
                  f"{draws_real} vs {draws_pred}",
                  help="El 1X2 rara vez marca el empate como más probable "
                       "(sesgo conocido); el corrector online ya infla "
                       "P(empate) del resto del torneo.")
        df_h = pd.DataFrame(rows_h).sort_values("Fecha", ascending=False)
        st.markdown(tbl(df_h, flags={"Local", "Visitante"}),
                    unsafe_allow_html=True)
        st.caption("Referencia honesta: 55-60% en 1X2 es el techo del "
                   "estado del arte; lo que importa es el Brier (calibración), "
                   "no el % de aciertos. Marcador exacto ~9-13%/partido, "
                   "así que ~20-25% con top-2 es excelente.")

# ------------------------------------------------ TAB: mercado
with tab_market:
    st.subheader("Modelo vs Mercado")
    st.caption("Las cuotas de las casas agregan información que el modelo no "
               "ve (lesiones de última hora, opinión experta) y calibran "
               "mejor los empates. Donde ingreses cuotas, el motor las mezcla "
               f"al {int(MARKET_WEIGHT * 100)}% en TODAS las predicciones "
               "(cards, eliminatorias, Monte Carlo).")

    if IS_ADMIN:
        with st.container(border=True):
            st.markdown('<h4 class="fc-title">➕ Ingresar cuotas 1X2</h4>',
                        unsafe_allow_html=True)
            pend = fixtures if pending.empty else pending
            oopts = {f"{r.date.date()} · {r.home_team} vs {r.away_team}": i
                     for i, r in pend.iterrows()}
            opk = st.selectbox("Partido", list(oopts), key="odds_pick")
            orow = pend.loc[oopts[opk]]
            oc1, oc2, oc3 = st.columns(3)
            oh = oc1.number_input(f"Cuota {orow.home_team}", 1.01, 100.0,
                                  2.00, 0.01, key="odd_h")
            od = oc2.number_input("Cuota empate", 1.01, 100.0, 3.40, 0.01,
                                  key="odd_d")
            oa = oc3.number_input(f"Cuota {orow.away_team}", 1.01, 100.0,
                                  3.80, 0.01, key="odd_a")
            if st.button("💾 Guardar cuotas", type="primary",
                         use_container_width=True):
                STORE.add_odds({"date": orow.date, "home_team": orow.home_team,
                                "away_team": orow.away_team, "odd_home": oh,
                                "odd_draw": od, "odd_away": oa})
                flash(f"💹 Cuotas guardadas: {orow.home_team} vs "
                      f"{orow.away_team}.")
                st.rerun()
        show_flash()

    odf = STORE.odds()
    if odf.empty:
        st.info("Aún no hay cuotas ingresadas. " + ("Agrégalas arriba para "
                "comparar modelo vs mercado." if IS_ADMIN else
                "El admin puede agregarlas."))
    else:
        rows_m = []
        for r in odf.itertuples(index=False):
            try:
                pa_m, pd_m, ph_m = devig(float(r.odd_home), float(r.odd_draw),
                                         float(r.odd_away))
            except (ValueError, ZeroDivisionError, TypeError):
                continue
            p = engine.predict_match(pd.Timestamp(r.date), r.home_team,
                                     r.away_team, r.home_team not in HOSTS)
            # valor: el modelo supera al mercado por >=6pp en algún resultado
            edges = {"Local": p["p_home"] - ph_m, "Empate": p["p_draw"] - pd_m,
                     "Visita": p["p_away"] - pa_m}
            best = max(edges, key=edges.get)
            val = (f"📈 {best} (+{100 * edges[best]:.0f}pp)"
                   if edges[best] >= 0.06 else "—")
            rows_m.append({
                "Local": r.home_team, "Visitante": r.away_team,
                "Modelo (L/X/V)": f"{100 * p['p_home']:.0f} / "
                f"{100 * p['p_draw']:.0f} / {100 * p['p_away']:.0f}",
                "Mercado (L/X/V)": f"{100 * ph_m:.0f} / {100 * pd_m:.0f} / "
                f"{100 * pa_m:.0f}",
                "Valor del modelo": val,
            })
        st.markdown(tbl(pd.DataFrame(rows_m),
                        flags={"Local", "Visitante"}),
                    unsafe_allow_html=True)
        st.caption("'Valor del modelo' = donde el modelo asigna ≥6pp más que "
                   "el mercado a un resultado. NO es consejo de apuesta: el "
                   "mercado suele tener razón, pero ahí está su desacuerdo.")


# ------------------------------------------------ TAB 3: líderes
with tab_leaders:
    st.markdown(
        '<div class="form-card"><h4>🏅 Líderes del torneo</h4>'
        '<p style="color:var(--muted);font-size:.82rem;margin:0">Estadísticas '
        'en vivo. Los datos se actualizan al ingresar cada resultado.</p></div>',
        unsafe_allow_html=True)
    pl = STORE.players()
    cards_df = STORE.discipline()
    if pl.empty and cards_df.empty:
        st.info("Aún no hay eventos ingresados. Registra goleadores y "
                "asistencias al guardar cada resultado.")
    else:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown('<div class="leader-badge">⚽ Goleadores</div>',
                        unsafe_allow_html=True)
            g = pl[pl.event.isin(["goal", "penalty"])]
            if g.empty:
                st.caption("Sin goles registrados.")
            else:
                top = (g.groupby(["player", "team"]).agg(
                        Goles=("event", "size"),
                        Penales=("event", lambda s: int((s == "penalty").sum())))
                       .reset_index()
                       .sort_values(["Goles", "Penales"],
                                    ascending=[False, True])
                       .head(10))
                for rank, (_, r) in enumerate(top.iterrows(), 1):
                    st.markdown(
                        f'<div class="leader-card" style="margin-bottom:8px">'
                        f'<div class="leader-num">{rank}</div>'
                        f'<div class="leader-name">{esc(r.player)}</div>'
                        f'<div class="leader-team">{flag_img(r.team, 16)} '
                        f'{esc(r.team)}</div>'
                       f'<div class="leader-stat">{int(r.Goles)} goles'
                       f'{" · " + str(int(r.Penales)) + " penal" if r.Penales else ""}'
                       f'</div></div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="leader-badge">🎯 Asistencias</div>',
                        unsafe_allow_html=True)
            a = pl[pl.event == "assist"]
            if a.empty:
                st.caption("Sin asistencias registradas.")
            else:
                top = (a.groupby(["player", "team"]).size()
                       .reset_index(name="Asistencias")
                       .sort_values("Asistencias", ascending=False)
                       .head(10))
                for rank, (_, r) in enumerate(top.iterrows(), 1):
                    st.markdown(
                        f'<div class="leader-card" style="margin-bottom:8px">'
                        f'<div class="leader-num">{rank}</div>'
                        f'<div class="leader-name">{esc(r.player)}</div>'
                        f'<div class="leader-team">{flag_img(r.team, 16)} '
                        f'{esc(r.team)}</div>'
                        f'<div class="leader-stat">{int(r.Asistencias)} asistencias'
                        f'</div></div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="leader-badge">🟨 Tarjetas</div>',
                        unsafe_allow_html=True)
            if cards_df.empty:
                st.caption("Sin tarjetas registradas.")
            else:
                top = (cards_df.groupby(["player", "team"]).agg(
                        Amarillas=("card", lambda s: int((s == "yellow").sum())),
                        Rojas=("card", lambda s: int((s == "red").sum())))
                       .reset_index()
                       .sort_values(["Rojas", "Amarillas"], ascending=False)
                       .head(10))
                for rank, (_, r) in enumerate(top.iterrows(), 1):
                    st.markdown(
                        f'<div class="leader-card" style="margin-bottom:8px">'
                        f'<div class="leader-num">{rank}</div>'
                        f'<div class="leader-name">{esc(r.player)}</div>'
                        f'<div class="leader-team">{flag_img(r.team, 16)} '
                        f'{esc(r.team)}</div>'
                        f'<div class="leader-stat">'
                        f'{"🟨 " * int(r.Amarillas)}{"🔴" * int(r.Rojas)}'
                        f'</div></div>', unsafe_allow_html=True)

# ------------------------------------------------ TAB 4: cuadro (bracket)
with tab_bracket:
    c1, c2 = st.columns([3, 1])
    c1.subheader("Cuadro del Mundial")
    n_sims_b = c2.selectbox("Simulaciones", [2000, 5000, 10000], index=1,
                            key="nsims_bracket")
    st.caption("Camino más probable (determinista): en cada llave avanza el "
               "equipo con P(avanza) > 50% en ESE cruce — los porcentajes de "
               "cada lado son justamente esa probabilidad del cruce, no la "
               "frecuencia Monte Carlo. En rojo, el que avanza; los cruces "
               "ya ingresados quedan al 100%. En el modo foco verás también "
               "los candidatos alternativos a ocupar cada llave.")
    with st.spinner("Simulando el torneo..."):
        payload = build_bracket_payload(STORE.token(), n_sims_b)
    render_bracket(payload, height=1050)

# ------------------------------------------------ TAB 5: eliminatorias
with tab_ko:
    st.subheader("Predictor de cruces")
    st.caption("Cuando se definan los cruces, elige las dos selecciones. "
               "P(avanza) incluye prórroga/penales aproximados por Elo.")
    st.caption("⚙️ **Modelo:** ensemble calibrado (Logística + Poisson "
               "Dixon-Coles) sobre Elo/forma/H2H/valor de plantilla "
               "pre-partido, con ajustes en vivo (momentum, suspensiones, "
               "lesiones) y corrección bayesiana del torneo.")
    teams48 = sorted(load_teams().name_canonical)
    c1, c2, c3 = st.columns([2, 2, 1])
    k1 = c1.selectbox("Equipo 1", teams48, key="k1")
    k2 = c2.selectbox("Equipo 2", teams48, index=1, key="k2")
    kdate = c3.date_input("Fecha del partido", pd.Timestamp("2026-06-28"),
                          key="kdate")
    if k1 != k2:
        neutral = k1 not in HOSTS
        p = engine.predict_match(pd.Timestamp(kdate), k1, k2, neutral)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Gana {k1}", f"{100 * p['p_home']:.0f}%")
        c2.metric("Empate (90')", f"{100 * p['p_draw']:.0f}%")
        c3.metric(f"Gana {k2}", f"{100 * p['p_away']:.0f}%")
        c1.metric(f"{k1} avanza", f"{100 * p['p_home_advances']:.0f}%")
        c2.metric("Goles esperados",
                  f"{p['lambda_home']:.1f} – {p['lambda_away']:.1f}")
        c3.metric(f"{k2} avanza", f"{100 * p['p_away_advances']:.0f}%")
        top5 = p["scorelines"][:5]
        pmax = top5[0][1] if top5 else 1.0
        cards = "".join(
            f'<div class="sl-card{" top" if i == 0 else ""}">'
            f'<div class="sl-tag">{"🎯 más probable" if i == 0 else f"#{i + 1}"}'
            f'</div><div class="sl-score">{s.replace("-", " – ")}</div>'
            f'<div class="sl-pct">{100 * pr:.1f}%</div>'
            f'<div class="sl-bar"><div style="width:{100 * pr / pmax:.0f}%">'
            f'</div></div></div>'
            for i, (s, pr) in enumerate(top5))
        chips = f'<div class="sl-grid">{cards}</div>'
        # señal de valor: si empate + no-favorito superan al favorito, el
        # favorito NO llega ni al 50% — el empate está bien cotizado
        ph_, pd_, pa_ = p["p_home"], p["p_draw"], p["p_away"]
        fav, under = (k1, k2) if ph_ >= pa_ else (k2, k1)
        fav_p, under_p = max(ph_, pa_), min(ph_, pa_)
        if pd_ + under_p > fav_p:
            chips += (f'<div class="bet-hint">💡 <b>Señal de valor:</b> '
                      f'empate ({100 * pd_:.0f}%) + {esc(under)} '
                      f'({100 * under_p:.0f}%) = '
                      f'{100 * (pd_ + under_p):.0f}% supera a {esc(fav)} '
                      f'({100 * fav_p:.0f}%) — el favorito no llega al 50%: '
                      f'el empate es una apuesta con valor.</div>')

        def xai_side(team: str) -> str:
            e = engine.state.explain(team, pd.Timestamp(kdate))
            parts = [f"momentum {e['momentum']:+.1f}"]
            parts += [f"{lbl} ({pts:+.0f})" for lbl, pts in e["items"]]
            return (f'<div class="mc-meta" style="margin-top:6px">'
                    f'{esc(team).upper()} · AJUSTE {e["total"]:+.1f} ELO · '
                    f'{esc(" · ".join(parts))}</div>')

        st.markdown(f'<div class="glass"><div class="mc-meta">MARCADORES MÁS '
                    f'PROBABLES</div>{chips}'
                    f'<div class="mc-meta" style="margin-top:10px">ELO '
                    f'EFECTIVO · {k1} {p["elo_home"]:.0f} · {k2} '
                    f'{p["elo_away"]:.0f}'
                    + ("" if neutral else f" · {k1} JUEGA EN CASA")
                    + '</div>' + xai_side(k1) + xai_side(k2)
                    + '</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Calendario de eliminatorias")
    kof = pd.DataFrame(load_ko_raw())
    kof = kof[kof["round"] != "Match for third place"]
    kof = kof[["round", "date", "team1", "team2", "ground"]].rename(columns={
        "round": "Ronda", "date": "Fecha", "team1": "Llave 1",
        "team2": "Llave 2", "ground": "Sede"})
    st.markdown(tbl(kof, height=420), unsafe_allow_html=True)

# ------------------------------------------------ TAB 6: camino al título
with tab_champ:
    c1, c2 = st.columns([3, 1])
    c1.subheader("Camino al título — Monte Carlo del torneo completo")
    n_sims = c2.selectbox("Simulaciones", [2000, 5000, 10000], index=1)
    st.caption("Simula el Mundial completo N veces desde el estado actual: "
               "grupos con marcadores muestreados del modelo (o el resultado "
               "real si ya lo ingresaste), desempates FIFA, mejores 8 "
               "terceros, bracket oficial y prórroga/penales por Elo. "
               "Se recalcula con cada resultado que ingreses.")

    with st.spinner(f"Simulando {n_sims:,} torneos..."):
        simdf, _ = run_simulation(STORE.token(), n_sims)

    podio = simdf.head(3)
    cols = st.columns(3)
    medallas = ["🥇 Favorito", "🥈 Segundo", "🥉 Tercero"]
    for c, med, (_, r) in zip(cols, medallas, podio.iterrows()):
        c.markdown(
            f'<div class="glass" style="text-align:center">'
            f'<div class="mc-meta">{med}</div>'
            f'<div style="margin-bottom:6px">{flag_img(r.team, 60)}</div>'
            f'<div style="font-weight:800;font-size:1.15rem;'
            f'color:var(--text)">{esc(r.team)}</div>'
            f'<div class="podium-pct">{100 * r.CAMPEON:.1f}%</div>'
            f'<div class="podium-lbl">Campeón</div>'
            f'<div class="mc-probs" style="justify-content:center;gap:14px">'
            f'<span>final {100 * r.F:.0f}%</span>'
            f'<span>semis {100 * r.SF:.0f}%</span></div>'
            f'</div>', unsafe_allow_html=True)

    show = simdf.rename(columns={
        "team": "Selección", "group": "Grupo", "R32": "Dieciseisavos",
        "R16": "Octavos", "QF": "Cuartos", "SF": "Semifinal",
        "F": "Final", "CAMPEON": "Campeón"})
    st.markdown(tbl(show, flags={"Selección"},
                    bars={"Dieciseisavos", "Octavos", "Cuartos", "Semifinal",
                          "Final", "Campeón"}, height=560),
                unsafe_allow_html=True)

# ------------------------------------------------ TAB 7: tablas
with tab_tablas:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Tablas de grupos")
        glive = live.merge(
            fixtures[["home_team", "away_team", "group"]],
            on=["home_team", "away_team"], how="inner") if len(live) else \
            pd.DataFrame(columns=["group"])
        if glive.empty:
            st.info("Aún no hay resultados de fase de grupos ingresados.")
        else:
            for g in sorted(glive.group.dropna().unique()):
                gm = glive[glive.group == g]
                stats: dict[str, list] = {}
                for r in gm.itertuples(index=False):
                    for team, gf, ga in ((r.home_team, r.home_score, r.away_score),
                                         (r.away_team, r.away_score, r.home_score)):
                        s = stats.setdefault(team, [0, 0, 0, 0, 0, 0, 0])
                        s[0] += 1                      # PJ
                        s[1] += gf > ga
                        s[2] += gf == ga
                        s[3] += gf < ga
                        s[4] += gf
                        s[5] += ga
                        s[6] += 3 * (gf > ga) + (gf == ga)
                t = (pd.DataFrame.from_dict(
                        stats, orient="index",
                        columns=["PJ", "G", "E", "P", "GF", "GC", "Pts"])
                     .assign(DG=lambda d: d.GF - d.GC)
                     .sort_values(["Pts", "DG", "GF"], ascending=False)
                     .reset_index(names="Selección"))
                st.markdown(f"**Grupo {g}**")
                st.markdown(tbl(t[["Selección", "PJ", "G", "E", "P", "GF",
                                   "GC", "DG", "Pts"]],
                                flags={"Selección"}),
                            unsafe_allow_html=True)
    with c2:
        st.subheader("Ranking Elo (48 clasificados)")
        rk = engine.elo_ranking(sorted(load_teams().name_canonical))
        rk["elo"] = rk["elo"].round(0).astype(int)
        rk = rk.reset_index(names="#").rename(columns={"team": "Selección",
                                                       "elo": "Elo"})
        st.markdown(tbl(rk, flags={"Selección"}, height=600),
                    unsafe_allow_html=True)

# ------------------------------------------------ TAB 8: auditoría de modelos
with tab_audit:
    st.subheader("Auditoría de modelos — backtesting")
    st.caption("Predicción PRE-partido reconstruida con anti-leakage y el "
               "mismo ensemble de producción (Logística + Poisson "
               "Dixon-Coles, pesos calibrados en validación). Los partidos "
               "del Mundial ingresados en vivo aparecen marcados 🏆 con la "
               "predicción exacta que el modelo hizo antes de conocer el "
               "resultado. ✓ = el real coincidió con el 1X2 más probable.")
    teams_all = sorted(load_teams().name_canonical)
    eq = st.selectbox("Selección a auditar", teams_all,
                      index=teams_all.index("Argentina")
                      if "Argentina" in teams_all else 0)
    bt = backtest_last(eq)
    if bt.empty:
        st.info("No hay partidos jugados con features completas para esta "
                "selección.")
    else:
        hits = int(bt.ok.sum())
        st.markdown(f'<div class="mc-meta" style="margin:6px 0 12px 0">'
                    f'ÚLTIMOS {len(bt)} PARTIDOS · MODELO ACERTÓ EL 1X2 EN '
                    f'{hits}/{len(bt)}</div>', unsafe_allow_html=True)
        lbl = {"H": "GANA LOCAL", "D": "EMPATE", "A": "GANA VISITA"}
        for r in bt.itertuples(index=False):
            cond = "LOCAL" if r.es_local else "VISITA"
            mark = " 🏆" if getattr(r, "live", False) else ""
            st.markdown(
                f'<div class="glass audit-row">'
                f'<span class="ar-date">{r.date.strftime("%d %b %Y")}{mark}'
                f'</span>'
                f'<span class="ar-rival">{flag_img(r.rival, 26)} '
                f'vs {esc(r.rival)} '
                f'<span class="ar-cond">({cond})</span></span>'
                f'<span class="ar-prob">{100 * r.p_win:.0f}%'
                f'<small>victoria {esc(eq)} (pre-partido)</small></span>'
                f'<span class="ar-score">{r.score}</span>'
                f'<span class="ar-prob" style="min-width:150px">'
                f'{"✓" if r.ok else "✗"} {lbl[r.pred]}'
                f'<small>pronóstico vs real: {lbl[r.real].lower()}</small>'
                f'</span>'
                f'<span class="ar-hit">{"✅" if r.ok else "❌"}</span>'
                f'</div>', unsafe_allow_html=True)
        st.caption("Recordatorio honesto: el techo del estado del arte en "
                   "1X2 internacional es ~55-60% de acierto; el valor real "
                   "del modelo está en su calibración (cuando dice 80%, "
                   "acierta ~80% de esas veces).")

    with st.expander("📈 Calibración global del modelo (test 2022+)"):
        cal = calibration_table()
        st.caption("Cada punto: partidos donde el modelo dio esa confianza "
                   "a su pronóstico vs cuántos acertó de verdad. La línea "
                   "'ideal' es calibración perfecta — pegado a ella = las "
                   "probabilidades significan lo que dicen.")
        chart = cal.set_index((100 * cal.confianza).round(0).astype(int))
        chart = pd.DataFrame({
            "% acierto real": 100 * chart.acierto,
            "ideal": chart.index.to_series()})
        st.line_chart(chart, x_label="confianza del modelo (%)",
                      y_label="acierto real (%)")

# ------------------------------------------------ TAB 0: la definición
with tab_final:
    _fd = build_final_payload(STORE.token())
    if _fd is None:
        st.info("Esta pestaña se enciende al cargar las dos semifinales: "
                "con solo la Final y el 3.er puesto pendientes, el espacio "
                "de desenlaces es exacto (sin Monte Carlo).")
    else:
        (_fa, _fb), (_ta, _tb) = _fd["finalists"], _fd["third"]
        _pod = _fd["podium"].set_index("team")

        st.markdown(
            '<div class="form-card"><h4>🏆 La definición del Mundial</h4>'
            '<p style="color:var(--muted);font-size:.82rem;margin:0">'
            'Quedan dos partidos. Aquí no hay Monte Carlo: probabilidades '
            'EXACTAS del ensemble (con correcciones online, mercado y '
            'prórroga/penales por Elo) sobre los únicos desenlaces '
            'posibles.</p></div>', unsafe_allow_html=True)

        # ---- cards de los 2 partidos que quedan
        def _fin_card(view: dict, title: str) -> str:
            _h, _a = view["teams"]
            _p = view["p"]
            _adv_h, _adv_a = _p["p_home_advances"], _p["p_away_advances"]
            def _side(team, adv, best):
                _w = "font-weight:800" if best else "opacity:.85"
                return (f'<div style="display:flex;justify-content:'
                        f'space-between;align-items:center;{_w};'
                        f'font-size:1.05rem;margin:4px 0">'
                        f'<span>{flag_img(team, 26)} {esc(team)}</span>'
                        f'<span>{100 * adv:.0f}%</span></div>')
            if view["real"]:
                _win = view["real"]["winner"]
                _p_pre = _adv_h if _win == _h else _adv_a
                _body = (
                    f'<div style="text-align:center;margin:8px 0">'
                    f'<div style="font-size:1.7rem;font-weight:900">'
                    f'{flag_img(_h, 26)} {view["real"]["score"]} '
                    f'{flag_img(_a, 26)}</div>'
                    f'<div style="font-weight:800;color:var(--accent)">'
                    f'Ganó {esc(_win)}</div>'
                    f'<div class="mc-meta">el modelo le daba '
                    f'{100 * _p_pre:.0f}%</div></div>')
            else:
                _sc, _psc = _p["score_pred"]
                _body = (
                    _side(_h, _adv_h, _adv_h >= _adv_a)
                    + '<div style="opacity:.5;text-align:center;'
                      'font-size:.7rem">vs</div>'
                    + _side(_a, _adv_a, _adv_a > _adv_h)
                    + f'<div class="mc-meta" style="margin-top:8px">90&prime;: '
                      f'{esc(_h)} {100 * _p["p_home"]:.0f}% · empate '
                      f'{100 * _p["p_draw"]:.0f}% · {esc(_a)} '
                      f'{100 * _p["p_away"]:.0f}%</div>'
                    + f'<div class="mc-meta">goles esperados '
                      f'{_p["lambda_home"]:.2f} – {_p["lambda_away"]:.2f} · '
                      f'si {"hay empate a 90&prime;" if _p["pred"] == "Empate" else "gana " + esc(_p["pred"])}, '
                      f'lo más probable: {_sc.replace("-", "–")} '
                      f'({100 * _psc:.0f}%)</div>')
            return (f'<div class="glass" style="padding:16px">'
                    f'<div class="mc-meta">{title} · {view["date"]} · '
                    f'{esc(view["ground"])}</div>{_body}</div>')

        _c1, _c2 = st.columns(2)
        _c1.markdown(_fin_card(_fd["final"], "GRAN FINAL"),
                     unsafe_allow_html=True)
        _c2.markdown(_fin_card(_fd["third_match"], "TERCER PUESTO"),
                     unsafe_allow_html=True)

        _k1, _k2, _k3, _k4 = st.columns(4)
        _k1.metric(f"{_fa} campeón", f"{100 * _pod.loc[_fa, 'p1']:.0f}%")
        _k2.metric(f"{_fb} campeón", f"{100 * _pod.loc[_fb, 'p1']:.0f}%")
        _k3.metric("Final a prórroga/penales",
                   f"{100 * _fd['final']['p']['p_draw']:.0f}%",
                   help="P(empate a los 90'). Si pasa, el desempate se "
                        "aproxima por Elo comprimido (TIEBREAK_DAMP).")
        _k4.metric("Aciertos 1X2 del modelo en el torneo",
                   f"{_fd['hits_audit']}/{_fd['n_audit']}",
                   f"{100 * _fd['hits_audit'] / max(_fd['n_audit'], 1):.0f}%")

        # ---- podio: matriz equipo × puesto (exacta, filas suman 1)
        st.subheader("Probabilidades de podio")
        _show = _fd["podium"].rename(columns={
            "team": "Selección", "p1": "Campeón", "p2": "Subcampeón",
            "p3": "3.er puesto", "p4": "4.º puesto"})
        st.markdown(tbl(_show, flags={"Selección"},
                        bars={"Campeón", "Subcampeón", "3.er puesto",
                              "4.º puesto"}), unsafe_allow_html=True)

        # ---- los 4 desenlaces posibles con probabilidad conjunta
        st.subheader("Los 4 desenlaces posibles")
        st.caption("Final ⊗ tercer puesto (independientes): el espacio "
                   "completo de podios, ordenado por probabilidad.")
        _mets = ("🥇", "🥈", "🥉", "4.º")
        _sc_cols = st.columns(4)
        for _col, _s in zip(_sc_cols, _fd["scenarios"]):
            _rows_html = "".join(
                f'<div style="display:flex;gap:6px;align-items:center;'
                f'margin:3px 0;font-size:.86rem">'
                f'<span style="width:24px">{_m}</span>'
                f'{flag_img(_t, 18)} <span>{esc(_t)}</span></div>'
                for _m, _t in zip(_mets, _s["podium"]))
            _col.markdown(
                f'<div class="glass" style="padding:14px">'
                f'<div class="podium-pct">{100 * _s["p"]:.1f}%</div>'
                f'{_rows_html}</div>', unsafe_allow_html=True)

        # ---- la final bajo el microscopio (matriz de marcadores exacta)
        st.subheader("La final bajo el microscopio")

        def _heat(view: dict) -> str:
            _m = view["matrix"]
            _pmax = max(max(_r) for _r in _m) or 1.0
            _h, _a = view["teams"]
            _head = "".join(f'<th style="font-weight:600">{_j}</th>'
                            for _j in range(6))
            _body = []
            for _i in range(6):
                _cells = []
                for _j in range(6):
                    _pv = _m[_i][_j]
                    _mix = int(round(86 * _pv / _pmax))
                    _lbl = f"{100 * _pv:.1f}" if _pv >= 0.005 else "·"
                    _cells.append(
                        f'<td style="background:color-mix(in srgb, '
                        f'var(--accent) {_mix}%, transparent);'
                        f'text-align:center;padding:7px 2px;'
                        f'border-radius:6px">{_lbl}</td>')
                _body.append(f'<tr><th style="font-weight:600">{_i}</th>'
                             + "".join(_cells) + "</tr>")
            return (f'<div class="glass" style="padding:14px">'
                    f'<div class="mc-meta">P(marcador exacto) % · filas '
                    f'{flag_img(_h, 16)} · columnas {flag_img(_a, 16)}</div>'
                    f'<div style="overflow-x:auto"><table style="width:100%;'
                    f'border-collapse:separate;border-spacing:3px;'
                    f'font-size:.74rem;color:var(--text)">'
                    f'<thead><tr><th></th>{_head}</tr></thead>'
                    f'<tbody>{"".join(_body)}</tbody></table></div></div>')

        _mc1, _mc2 = st.columns([1.15, 1])
        _mc1.markdown(_heat(_fd["final"]), unsafe_allow_html=True)
        with _mc2:
            _pfin = _fd["final"]["p"]
            _top5 = _pfin["scorelines"][:5]
            _pmax5 = _top5[0][1] if _top5 else 1.0
            _chips = "".join(
                f'<div class="sl-card{" top" if _i == 0 else ""}">'
                f'<div class="sl-tag">'
                f'{"🎯 más probable" if _i == 0 else f"#{_i + 1}"}</div>'
                f'<div class="sl-score">{_s.replace("-", " – ")}</div>'
                f'<div class="sl-pct">{100 * _pr:.1f}%</div>'
                f'<div class="sl-bar">'
                f'<div style="width:{100 * _pr / _pmax5:.0f}%"></div></div>'
                f'</div>'
                for _i, (_s, _pr) in enumerate(_top5))
            st.markdown(
                f'<div class="glass"><div class="mc-meta">MARCADORES MÁS '
                f'PROBABLES · {esc(_fa)} – {esc(_fb)}</div>'
                f'<div class="sl-grid">{_chips}</div>'
                f'<div class="mc-meta" style="margin-top:10px">'
                f'Ambos marcan {100 * _fd["final"]["p_btts"]:.0f}% · '
                f'Más de 2.5 goles {100 * _fd["final"]["p_over25"]:.0f}% · '
                f'Elo efectivo {esc(_fa)} {_pfin["elo_home"]:.0f} / '
                f'{esc(_fb)} {_pfin["elo_away"]:.0f}</div></div>',
                unsafe_allow_html=True)

        # ---- Bota de Oro
        st.subheader("Carrera por la Bota de Oro")
        _race = _fd["race"]
        if len(_race):
            _rshow = _race.rename(columns={
                "player": "Jugador", "team": "Selección", "goals": "Goles",
                "assists": "Asist.", "lam": "xG restante",
                "p_top_solo": "Bota en solitario",
                "p_top_shared": "Al menos compartida"})
            _rshow["xG restante"] = _rshow["xG restante"].map(
                lambda x: f"{x:.2f}")
            st.markdown(tbl(_rshow[["Jugador", "Selección", "Goles",
                                    "Asist.", "xG restante",
                                    "Bota en solitario",
                                    "Al menos compartida"]],
                            flags={"Selección"},
                            bars={"Bota en solitario",
                                  "Al menos compartida"}),
                        unsafe_allow_html=True)
            st.caption("Goles restantes de cada jugador ~ Poisson(goles "
                       "esperados de su equipo en el partido que le queda × "
                       "su cuota de los goles del equipo en el torneo). "
                       "Aproximación declarada: jugadores independientes "
                       "(la correlación Kane–Bellingham se desprecia) y el "
                       "desempate oficial (asistencias, luego minutos) NO "
                       "se modela — las asistencias mostradas son las "
                       "registradas en la ingesta.")
        else:
            st.info("Sin goleadores registrados todavía.")

        # ---- cómo llegaron: el camino de los 4
        st.subheader("Cómo llegaron — el camino de los 4")
        st.caption("Cada línea muestra la predicción PRE-partido del modelo "
                   "(la misma que alimentó al corrector online, sin "
                   "recalcular) y si acertó el 1X2. 🟢 ganó · 🟡 empató "
                   "· 🔴 perdió · (des.) = avanzó por prórroga/penales.")
        _RES_DOT = {"G": "🟢", "E": "🟡", "P": "🔴"}
        for _pair, _rol in ((_fd["finalists"], "FINALISTA"),
                            (_fd["third"], "POR EL 3.ER PUESTO")):
            _jc = st.columns(2)
            for _col, _t in zip(_jc, _pair):
                _j = _fd["journeys"][_t]
                _lines = "".join(
                    f'<div style="display:flex;justify-content:'
                    f'space-between;gap:8px;align-items:center;'
                    f'margin:4px 0;font-size:.84rem">'
                    f'<span style="display:flex;gap:6px;align-items:center">'
                    f'<span class="mc-meta" style="width:52px">'
                    f'{esc(_st["stage"])}</span>'
                    f'{flag_img(_st["rival"], 16)} {esc(_st["rival"])}</span>'
                    f'<span style="white-space:nowrap">'
                    f'{_st["gf"]}–{_st["gc"]}'
                    f'{" (des.)" if _st["pens"] else ""} '
                    f'{_RES_DOT[_st["res"]]} '
                    f'<span class="mc-meta">{100 * _st["p_pre"]:.0f}%</span> '
                    f'{"✓" if _st["hit"] else "✗"}</span></div>'
                    for _st in _j["steps"])
                _sur = _j["surprise"]
                _sur_txt = (f'Su triunfo más improbable: {_sur[2]} a '
                            f'{esc(_sur[0])} (el modelo le daba '
                            f'{100 * _sur[1]:.0f}%).' if _sur else "")
                _extra = (f' · {_j["ko_extra"]} definido(s) en '
                          f'prórroga/penales' if _j["ko_extra"] else "")
                _col.markdown(
                    f'<div class="glass" style="padding:16px">'
                    f'<div style="display:flex;justify-content:'
                    f'space-between;align-items:center;margin-bottom:6px">'
                    f'<span style="font-weight:800;font-size:1.05rem">'
                    f'{flag_img(_t, 24)} {esc(_t)}</span>'
                    f'<span class="mc-meta">{_rol}</span></div>'
                    f'<div class="mc-meta" style="margin-bottom:8px">'
                    f'{_j["won"]}G-{_j["draw"]}E-{_j["lost"]}P · '
                    f'GF {_j["gf"]} / GC {_j["gc"]}{_extra} · Elo efectivo '
                    f'{_j["elo"]:.0f} (momentum {_j["momentum"]:+.0f}, '
                    f'ajuste total {_j["adjust"]:+.0f})</div>'
                    f'{_lines}'
                    f'<div class="mc-meta" style="margin-top:8px">'
                    f'El modelo acertó {_j["hits"]}/{len(_j["steps"])} de '
                    f'sus partidos. {_sur_txt}</div></div>',
                    unsafe_allow_html=True)

        st.caption("⚙️ **Metodología:** ensemble Logística + Poisson "
                   "Dixon-Coles con Elo/forma/H2H/valor de plantilla, "
                   "momentum del torneo (K=45), corrección bayesiana online "
                   "(goles, empates, altitud), mezcla con cuotas de mercado "
                   "donde existan y desempate de prórroga/penales por Elo "
                   "comprimido. Con las semifinales cargadas, campeón/"
                   "subcampeón dependen SOLO de la Final y 3.º/4.º SOLO del "
                   "tercer puesto: las probabilidades de esta pestaña son "
                   "exactas bajo el modelo, no frecuencias simuladas.")
