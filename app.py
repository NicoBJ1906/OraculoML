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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
import streamlit as st

from frontend import inject_effects, render_bracket
from mundial import auth
from mundial.live.engine import LiveEngine
from mundial.live.store import LiveStore
from mundial.predict.montecarlo import TournamentSimulator

STORE = LiveStore(ROOT)
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

/* ---- sidebar (RBAC) ---- */
section[data-testid="stSidebar"] {
  background: var(--surface-solid);
  border-right: 1px solid var(--border);}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] strong {color: var(--text) !important;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p
  {color: var(--muted) !important;}

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
.mc-team .flag img {filter: drop-shadow(0 4px 12px var(--shadow));}
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

/* ---- mc-probs heredado (podium Camino al título) ---- */
.mc-probs {display: flex; font-size: .78rem; font-weight: 600; color: var(--muted);}
.mc-probs span:first-child {color: var(--accent);}
.mc-probs span:last-child {color: var(--bar-a1);}
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

/* ---- checkboxes / toggles ---- */
div[data-testid="stCheckbox"] label span {font-weight: 600;}
.stToggle {gap: 8px;}
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
    return joblib.load(ROOT / "models" / "artifacts.joblib")


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
    return LiveEngine(matches, art["clf"], art["pois_home"],
                      art["pois_away"], art["rho"], art["blend"], STORE)


@st.cache_data
def run_simulation(live_tok: str, n_sims: int) -> tuple[pd.DataFrame, dict]:
    """Monte Carlo del torneo completo; se invalida al ingresar resultados."""
    eng = build_engine(live_tok)
    teams = load_teams()
    groups = teams.groupby("group")["name_canonical"].apply(list).to_dict()
    sim = TournamentSimulator(eng, load_fixtures(), STORE.results(),
                              load_ko_raw(), groups)
    df = sim.run(n_sims)
    return df, sim.slot_stats


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
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.checkbox("Registrar xG", key=f"{k}_usexg"):
            out["xg_home"] = c1.number_input(f"xG {home}", 0.0, 15.0, 1.0,
                                             0.1, key=f"{k}_xgh")
            out["xg_away"] = c2.number_input(f"xG {away}", 0.0, 15.0, 1.0,
                                             0.1, key=f"{k}_xga")
        w = c3.selectbox("Clima", WEATHER_OPTS, key=f"{k}_wx")
        if w != "Sin dato":
            out["weather"] = w
        c1, c2 = st.columns(2)
        fh = c1.text_input(f"Formación {home}", placeholder="4-3-3",
                           key=f"{k}_fh")
        fa = c2.text_input(f"Formación {away}", placeholder="4-4-2",
                           key=f"{k}_fa")
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
                st.markdown(f'<div class="mc-meta" style="margin-top:8px">'
                            f'Detalle</div>', unsafe_allow_html=True)
                for lbl, pts in expl["items"]:
                    st.markdown(
                        f'<div class="xai-stat"><span class="label">{esc(lbl)}</span>'
                        f'<span class="value">{pts:+.0f}</span></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="xai-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mc-meta">Correcciones globales del torneo</div>',
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
        st.markdown(f'<div class="mc-meta">Marcadores más probables</div>',
                    unsafe_allow_html=True)
        for s, pr in p["scorelines"]:
            st.markdown(f'<span class="pill">{s.replace("-", " – ")} · '
                        f'{100 * pr:.0f}%</span>', unsafe_allow_html=True)

    # ---- sección pedagógica: qué significa todo esto (para no expertos)
    with st.expander("📚 ¿Qué es el rating Elo y cómo afecta esta predicción?"):
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

hcol, tcol = st.columns([5, 1])
hcol.markdown('<h1 class="hero">Oráculo personal de Nicolás — '
              '<span class="grad">Mundial 2026</span></h1>',
              unsafe_allow_html=True)
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
auth.login_widget()
IS_ADMIN = auth.is_admin()
_labels = ["Próximos partidos", "Líderes", "Cuadro", "Eliminatorias",
           "Camino al título", "Tablas"]
if IS_ADMIN:
    _labels.insert(1, "Ingresar resultado")
_tabs = dict(zip(_labels, st.tabs(_labels)))
tab_pred = _tabs["Próximos partidos"]
tab_leaders = _tabs["Líderes"]
tab_bracket = _tabs["Cuadro"]
tab_ko = _tabs["Eliminatorias"]
tab_champ = _tabs["Camino al título"]
tab_tablas = _tabs["Tablas"]
tab_result = _tabs.get("Ingresar resultado")

# ------------------------------------------------ TAB 1: predicciones
with tab_pred:
    if pending.empty:
        st.info("No quedan fixtures de fase de grupos pendientes.")
    else:
        dias = st.slider("Días a mostrar", 1, 30, 4)
        d0 = pending.date.min()
        sel = pending[pending.date <= d0 + pd.Timedelta(days=dias - 1)]
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
                              key=f"xai_pred_{idx}"):
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

        with col_a:
            st.markdown('<div class="form-card"><h4>🏆 Fase de grupos</h4>',
                        unsafe_allow_html=True)
            if pending.empty:
                st.info("No hay fixtures pendientes de fase de grupos.")
                st.markdown('</div>', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="form-card"><h4>⚔️ Eliminatoria / manual</h4>',
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
            st.markdown('</div>', unsafe_allow_html=True)

        if len(live):
            st.markdown(f'<div class="form-card" style="margin-top:12px">'
                        f'<h4>📜 Resultados ingresados ({len(live)})</h4>',
                        unsafe_allow_html=True)
            show = live[["date", "home_team", "home_score", "away_score",
                         "away_team", "xg_home", "xg_away", "weather",
                         "stage"]].copy()
            show["date"] = pd.to_datetime(show["date"]).dt.date
            show = show.rename(columns={
                "date": "Fecha", "home_team": "Local", "home_score": "",
                "away_score": "", "away_team": "Visitante", "xg_home": "xG",
                "xg_away": "xG", "weather": "Clima", "stage": "Fase"})
            st.markdown(tbl(show.fillna("—").sort_values("Fecha",
                                                         ascending=False),
                            flags={"Local", "Visitante"}, height=280),
                        unsafe_allow_html=True)
            if st.button("🗑️ Borrar el último resultado"):
                STORE.delete_match(str(live.iloc[-1].match_id))
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

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
    st.caption("Filtra por fase con los botones. Cada llave muestra el "
               "ocupante más probable según el Monte Carlo y en rojo el "
               "favorito a avanzar; los cruces ya jugados quedan al 100%.")
    with st.spinner("Simulando el torneo..."):
        simdf, slots = run_simulation(STORE.token(), n_sims_b)

    def _flag_url(team: str, size: int = 40) -> str | None:
        iso = FLAG_ISO.get(team)
        return f"https://flagcdn.com/w{size}/{iso}.png" if iso else None

    def _side(cands: list) -> dict | None:
        """Lado de una llave según el contrato JSON del bracket (spec §7)."""
        if not cands:
            return None
        team, share = cands[0]
        return {"team": team, "flag": _flag_url(team),
                "pct": round(100 * share, 1)}

    ko = sorted(load_ko_raw(), key=lambda m: m.get("num", 999))
    ko = [m for m in ko if m["round"] != "Match for third place"]
    round_lbl = {"Round of 32": "Dieciseisavos", "Round of 16": "Octavos",
                 "Quarter-final": "Cuartos", "Semi-final": "Semifinal",
                 "Final": "Final"}

    rounds_payload, champion = [], None
    for rnd, label in round_lbl.items():
        matches = []
        for i, m in enumerate(ko):
            if m["round"] != rnd:
                continue
            ss = slots.get(str(m.get("num", f"x{i}")),
                           {"t1": [], "t2": [], "w": []})
            winner = ss["w"][0][0] if ss["w"] else None
            matches.append({
                "t1": _side(ss["t1"]), "t2": _side(ss["t2"]), "win": winner,
                "cands1": ss["t1"][:3], "cands2": ss["t2"][:3],
                "date": pd.Timestamp(m["date"]).strftime("%d %b").upper(),
                "ground": m.get("ground", "")})
            if rnd == "Final" and ss["w"]:
                champion = {"team": winner, "flag": _flag_url(winner),
                            "pct": round(100 * ss["w"][0][1], 1)}
        rounds_payload.append({"key": rnd.replace(" ", "_"),
                               "label": label, "matches": matches})

    render_bracket({"rounds": rounds_payload, "champion": champion},
                   height=790)

# ------------------------------------------------ TAB 5: eliminatorias
with tab_ko:
    st.subheader("Predictor de cruces")
    st.caption("Cuando se definan los cruces, elige las dos selecciones. "
               "P(avanza) incluye prórroga/penales aproximados por Elo.")
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
            f'<div class="mc-score" style="font-size:2rem">'
            f'{100 * r.CAMPEON:.1f}%<small>campeón</small></div>'
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
                        s[1] += gf > ga; s[2] += gf == ga; s[3] += gf < ga
                        s[4] += gf; s[5] += ga
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
