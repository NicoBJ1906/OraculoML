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
from mundial.live.engine import LiveEngine
from mundial.live.store import LiveStore
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

st.set_page_config(page_title="Oráculo personal de Nicolás — Mundial 2026",
                   page_icon="⚽", layout="wide")

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


@st.cache_resource(max_entries=2)
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


@st.cache_data
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
        if t1 and t2:
            win = ko_real.get(frozenset((t1, t2)))
            if win:
                pwin = 100
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
            "win": win, "pwin": pwin,
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
    if f.empty:
        return pd.DataFrame()
    p_clf = art["clf"].predict_proba(f[FEATURES])        # columnas A, D, H
    lh = art["pois_home"].predict(f[POISSON_FEATURES])
    la = art["pois_away"].predict(f[POISSON_FEATURES])
    rows = []
    for k, (_, r) in enumerate(f.iterrows()):
        pp = outcome_probs(score_matrix(float(lh[k]), float(la[k]),
                                        art["rho"]))
        p = (art["blend"] * p_clf[k]
             + (1 - art["blend"]) * np.array([pp["A"], pp["D"], pp["H"]]))
        pa, pd_, ph = float(p[0]), float(p[1]), float(p[2])
        pred = max((("H", ph), ("D", pd_), ("A", pa)), key=lambda x: x[1])[0]
        real = ("H" if r.home_score > r.away_score
                else "A" if r.away_score > r.home_score else "D")
        es_local = r.home_team == team
        rows.append({"date": pd.Timestamp(r.date),
                     "rival": r.away_team if es_local else r.home_team,
                     "es_local": es_local,
                     "p_win": ph if es_local else pa, "p_draw": pd_,
                     "score": f"{int(r.home_score)} – {int(r.away_score)}",
                     "pred": pred, "real": real, "ok": pred == real})
    return pd.DataFrame(rows).sort_values("date", ascending=False)


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
    cols = st.columns(widths, vertical_alignment="bottom")
    vals: dict = {}
    equipo_key = f"{key}_{nonce}_Equipo"
    for kk, (name, kind, opts) in enumerate(fields):
        wkey = f"{key}_{nonce}_{name}"
        if kind == "select":
            vals[name] = cols[kk].selectbox(name, opts, key=wkey)
        elif kind == "int":
            vals[name] = cols[kk].number_input(name, 1, 130, 1, key=wkey)
        elif kind == "player":
            # dropdown anti-typos: plantilla del equipo elegido en esta fila
            team_sel = st.session_state.get(equipo_key, opts[0] if opts else "")
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
    if cols[-1].button("＋", key=f"{key}_add{nonce}",
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
_labels = ["Próximos partidos", "Líderes", "Cuadro", "Eliminatorias",
           "Camino al título", "Tablas", "Auditoría"]
if IS_ADMIN:
    _labels.insert(1, "Ingresar resultado")
_tabs = dict(zip(_labels, st.tabs(_labels)))
tab_pred = _tabs["Próximos partidos"]
tab_leaders = _tabs["Líderes"]
tab_bracket = _tabs["Cuadro"]
tab_ko = _tabs["Eliminatorias"]
tab_champ = _tabs["Camino al título"]
tab_tablas = _tabs["Tablas"]
tab_audit = _tabs["Auditoría"]
tab_result = _tabs.get("Ingresar resultado")

# ------------------------------------------------ TAB 1: predicciones
with tab_pred:
    if pending.empty:
        st.info("No quedan fixtures de fase de grupos pendientes.")
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
                    st.success(f"{row.home_team} {gh} – {ga} {row.away_team} "
                               "guardado. Predicciones recalculadas.")
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
                    st.success("Partido guardado. Predicciones recalculadas.")
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
            if hist.button("🗑️ Borrar el último resultado"):
                STORE.delete_match(str(live.iloc[-1].match_id))
                st.rerun()

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
    render_bracket(payload, height=790)

# ------------------------------------------------ TAB 5: eliminatorias
with tab_ko:
    st.subheader("Predictor de cruces")
    st.caption("Cuando se definan los cruces, elige las dos selecciones. "
               "P(avanza) incluye prórroga/penales aproximados por Elo.")
    st.caption("⚙️ **Modelo:** ensemble Regresión Logística (peso 0.8) + "
               "Poisson Dixon-Coles (0.2, ρ=−0.15) sobre features "
               "Elo/forma/H2H pre-partido, con ajustes en vivo (momentum, "
               "suspensiones, lesiones) y corrección bayesiana del torneo "
               "(ritmo de goles, empates, altitud).")
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
        chips = "".join(f'<span class="pill">{s.replace("-", " – ")} · '
                        f'{100 * pr:.0f}%</span>'
                        for s, pr in p["scorelines"])

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
    st.caption("Predicción PRE-partido reconstruida desde la capa Gold "
               "(features con anti-leakage) con los artefactos entrenados — "
               "el mismo ensemble de producción: Regresión Logística (0.8) "
               "+ Poisson Dixon-Coles (0.2, ρ=−0.15). ✓ = el resultado real "
               "coincidió con el 1X2 más probable del modelo.")
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
            st.markdown(
                f'<div class="glass audit-row">'
                f'<span class="ar-date">{r.date.strftime("%d %b %Y")}'
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
