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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, p, span, div, label {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {background: var(--bg);}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.block-container {padding-top: 2.2rem; max-width: 1240px;}
section[data-testid="stMain"], .block-container {position: relative; z-index: 1;}

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

/* ---- hero ---- */
.hero {font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: 3.6rem; font-weight: 700; letter-spacing: -.04em;
  line-height: 1.04; color: var(--text); margin: 0 0 .35rem 0;}
.hero .grad {background: linear-gradient(92deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero-sub {color: var(--hero-sub); font-size: .95rem; margin: 0 0 .4rem 0;}
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
  backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 999px; padding: 5px; gap: 2px; width: fit-content;
  box-shadow: 0 8px 36px var(--glow);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 999px; padding: 7px 18px; font-weight: 600;
  background: transparent; color: var(--muted);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #fff !important;
  box-shadow: 0 4px 16px var(--glow);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]
  {display: none;}

/* ---- cards ---- */
.glass {
  background: var(--surface);
  backdrop-filter: blur(26px) saturate(160%);
  -webkit-backdrop-filter: blur(26px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: 0 12px 44px var(--shadow), 0 0 50px var(--glow);
  padding: 18px 20px; margin-bottom: 14px;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.glass:hover {
  transform: translateY(-3px);
  border-color: var(--accent);
}
.match-card {min-height: 172px;}
.mc-meta {font-size: .68rem; font-weight: 600; letter-spacing: .12em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 12px;}
.mc-row {display: flex; align-items: center; justify-content: space-between;}
.mc-team {flex: 1; text-align: center; font-weight: 700; font-size: .92rem;
          color: var(--text); line-height: 1.25;}
.mc-team .flag {display: block; margin-bottom: 6px;}
.mc-team .flag img {filter: drop-shadow(0 4px 12px var(--shadow));}
.mc-score {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700;
  padding: 0 10px; letter-spacing: -.02em; white-space: nowrap;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.mc-score small {display: block; font-size: .58rem; font-weight: 600;
                 -webkit-text-fill-color: var(--muted); text-transform: uppercase;
                 letter-spacing: .12em; text-align: center;}
.mc-bar {display: flex; height: 7px; border-radius: 999px; overflow: hidden;
         margin: 14px 0 7px 0; background: var(--bar-d);}
.mc-bar .h {background: linear-gradient(90deg, var(--accent), var(--accent2));}
.mc-bar .d {background: transparent;}
.mc-bar .a {background: linear-gradient(90deg, var(--bar-a1), var(--bar-a2));}
.mc-probs {display: flex; justify-content: space-between; font-size: .78rem;
           font-weight: 600; color: var(--muted);}
.mc-probs span:first-child {color: var(--accent);}
.mc-probs span:last-child {color: var(--bar-a1);}
.pill {display: inline-block; background: var(--surface);
       border-radius: 999px; padding: 4px 13px; margin: 2px 5px 2px 0;
       font-size: .8rem; font-weight: 600;
       border: 1px solid var(--border); color: var(--text);}
.mc-adj {display: inline-block; margin-left: 8px; padding: 1px 8px;
         border-radius: 999px; background: var(--surface);
         border: 1px solid var(--accent); color: var(--accent);
         font-size: .62rem; letter-spacing: .08em;}
.rowchip {background: var(--input-bg); border: 1px solid var(--border);
          border-radius: 10px; padding: 7px 12px; font-size: .84rem;
          color: var(--text); overflow: hidden; text-overflow: ellipsis;
          white-space: nowrap;}

/* ---- inputs y botones ---- */
.stButton > button {
  border-radius: 999px; font-weight: 700; border: none;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff; padding: .5rem 1.5rem;
  box-shadow: 0 6px 22px var(--glow);
  transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover {transform: translateY(-2px); color: #fff;}
div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input,
.stTextInput input {
  border-radius: 13px !important; background: var(--input-bg) !important;
  color: var(--text) !important; border-color: var(--border) !important;
}
div[data-baseweb="select"] svg {fill: var(--muted);}
div[data-baseweb="popover"] ul, ul[role="listbox"] {
  background: var(--surface-solid) !important;}
ul[role="listbox"] li {color: var(--text) !important;}
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px; padding: 14px 18px;
  box-shadow: 0 12px 40px var(--shadow);
}
div[data-testid="stMetricValue"] {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  color: var(--text);}
div[data-testid="stMetricLabel"] p {color: var(--muted) !important;}
div[data-testid="stExpander"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px;}
div[data-testid="stExpander"] summary {color: var(--text);}
div[data-testid="stSlider"] p {color: var(--muted);}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
  border-radius: 14px; overflow: hidden;
  border: 1px solid var(--border);}

/* ---- tablas HTML ---- */
.tblwrap {background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; overflow: auto; margin-bottom: 14px;
  box-shadow: 0 10px 36px var(--shadow);}
.tbl {width: 100%; border-collapse: collapse; font-size: .85rem;}
.tbl th {position: sticky; top: 0; background: var(--thead);
  backdrop-filter: blur(20px);
  color: var(--muted); text-transform: uppercase; font-size: .66rem;
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
  margin-right: 8px;}
.pbar div {height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));}

/* ---- bracket ---- */
.bracket {display: flex; gap: 14px; min-height: 1180px; margin-top: 8px;}
.b-col {flex: 1; display: flex; flex-direction: column;
        justify-content: space-around; gap: 6px;}
.b-round {text-align: center; font-size: .68rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  margin-bottom: 2px;}
.b-match {background: var(--surface); border: 1px solid var(--border);
  border-radius: 13px; padding: 7px 10px;
  box-shadow: 0 6px 22px var(--shadow);}
.b-side {display: flex; align-items: center; gap: 6px; font-size: .76rem;
  font-weight: 600; color: var(--text); padding: 3px 2px;
  white-space: nowrap; overflow: hidden;}
.b-side img {border-radius: 3px; flex-shrink: 0;}
.b-side .b-name {overflow: hidden; text-overflow: ellipsis; flex: 1;}
.b-side .b-p {color: var(--muted); font-size: .7rem; font-weight: 700;}
.b-side.b-win {color: var(--accent);}
.b-side.b-win .b-p {color: var(--accent);}
.b-foot {font-size: .62rem; color: var(--muted); margin-top: 2px;
  letter-spacing: .04em;}
.b-champ {background: linear-gradient(135deg, var(--accent), var(--accent2));
  border: none; text-align: center; padding: 14px 10px;}
.b-champ .b-side {justify-content: center; color: #fff;}
.b-champ .b-p {color: rgba(255,255,255,.85) !important;}
.b-champ .b-tag {font-size: .62rem; font-weight: 700; letter-spacing: .16em;
  color: rgba(255,255,255,.85); text-transform: uppercase;}
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


def match_card(r, p, adj: tuple[float, float] | None = None) -> str:
    """HTML de una card de partido con barra de probabilidades.
    adj: ajuste Elo en vivo (local, visitante) para el chip XAI."""
    fecha = pd.Timestamp(r.date).strftime("%a %d %b").upper()
    sede = f"{r.city}" + ("" if r.neutral else " · CASA")
    grupo = f"GRUPO {r.group} · " if r.group else ""
    ph, pd_, pa = p["p_home"], p["p_draw"], p["p_away"]
    score = p["score_pred"][0].replace("-", " – ")
    adj_html = ""
    if adj and (abs(adj[0]) >= 0.5 or abs(adj[1]) >= 0.5):
        adj_html = (f'<div class="mc-adj">Δ EN VIVO · {adj[0]:+.0f} ELO'
                    f' / {adj[1]:+.0f} ELO</div>')
    return (
        f'<div class="glass match-card">'
        f'<div class="mc-meta">{fecha} · {grupo}{sede}{adj_html}</div>'
        f'<div class="mc-row">'
        f'<div class="mc-team"><span class="flag">{flag_img(r.home_team)}</span>'
        f'{esc(r.home_team)}</div>'
        f'<div class="mc-score">{score}<small>pred.</small></div>'
        f'<div class="mc-team"><span class="flag">{flag_img(r.away_team)}</span>'
        f'{esc(r.away_team)}</div>'
        f'</div>'
        f'<div class="mc-bar">'
        f'<div class="h" style="width:{100 * ph:.0f}%"></div>'
        f'<div class="d" style="width:{100 * pd_:.0f}%"></div>'
        f'<div class="a" style="width:{100 * pa:.0f}%"></div>'
        f'</div>'
        f'<div class="mc-probs"><span>{100 * ph:.0f}%</span>'
        f'<span>empate {100 * pd_:.0f}%</span><span>{100 * pa:.0f}%</span></div>'
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
    for kk, (name, kind, opts) in enumerate(fields):
        wkey = f"{key}_{nonce}_{name}"
        if kind == "select":
            vals[name] = cols[kk].selectbox(name, opts, key=wkey)
        elif kind == "int":
            vals[name] = cols[kk].number_input(name, 1, 130, 1, key=wkey)
        else:
            vals[name] = cols[kk].text_input(name, key=wkey,
                                             placeholder="Nombre")
    if cols[-1].button("＋", key=f"{key}_add{nonce}",
                       help="Agregar fila"):
        if all(str(vals[n]).strip() for n, kind, _ in fields
               if kind == "text"):
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
            ("Equipo", "select", teams), ("Jugador", "text", None),
            ("Minuto", "int", None), ("Tipo", "select", list(EVENT_LABELS))])
        cards = dynamic_rows("Tarjetas", f"rows_cd_{prefix}", [
            ("Equipo", "select", teams), ("Jugador", "text", None),
            ("Tarjeta", "select", list(CARD_LABELS)),
            ("Minuto", "int", None)])
        inj = dynamic_rows("Lesiones de jugadores clave", f"rows_in_{prefix}",
                           [("Equipo", "select", teams),
                            ("Jugador", "text", None),
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

(tab_pred, tab_result, tab_leaders, tab_bracket, tab_ko, tab_champ,
 tab_tablas) = st.tabs(
    ["Próximos partidos", "Ingresar resultado", "Líderes", "Cuadro",
     "Eliminatorias", "Camino al título", "Tablas"])

# ------------------------------------------------ TAB 1: predicciones
with tab_pred:
    if pending.empty:
        st.info("No quedan fixtures de fase de grupos pendientes.")
    else:
        dias = st.slider("Días a mostrar", 1, 30, 4)
        d0 = pending.date.min()
        sel = pending[pending.date <= d0 + pd.Timedelta(days=dias - 1)]
        cards = []
        for r in sel.itertuples(index=False):
            p = engine.predict_match(r.date, r.home_team, r.away_team,
                                     r.neutral, city=r.city)
            adj = (engine.state.adjustment(r.home_team, r.date),
                   engine.state.adjustment(r.away_team, r.date))
            cards.append(match_card(r, p, adj))
        for i in range(0, len(cards), 3):
            cols = st.columns(3)
            for c, html in zip(cols, cards[i:i + 3]):
                c.markdown(html, unsafe_allow_html=True)
        st.caption("Barra: rojo = local · centro = empate · azul = visitante. "
                   "El marcador es el más probable según el Poisson. CASA = "
                   "anfitrión en su país. Las predicciones ya incluyen "
                   "momentum, sanciones, lesiones y corrección online.")

        # ---- XAI: desglose de los modificadores en vivo
        with st.expander("¿Por qué? — Desglose de ajustes en vivo (XAI)"):
            next_date = {}
            for r in pending.itertuples(index=False):
                next_date.setdefault(r.home_team, r.date)
                next_date.setdefault(r.away_team, r.date)
            xai_rows = []
            for team, d in sorted(next_date.items()):
                e = engine.state.explain(team, d)
                if abs(e["total"]) < 0.5:
                    continue
                detalle = " · ".join(f"{lbl} ({pts:+.0f})"
                                     for lbl, pts in e["items"]) or "—"
                xai_rows.append({
                    "Selección": team,
                    "Ajuste total": f'{e["total"]:+.1f}',
                    "Momentum": f'{e["momentum"]:+.1f}',
                    "Sanciones y lesiones": detalle})
            if xai_rows:
                st.markdown(tbl(pd.DataFrame(xai_rows),
                                flags={"Selección"}, height=380),
                            unsafe_allow_html=True)
            else:
                st.caption("Sin modificadores activos: aún no hay momentum, "
                           "sanciones ni lesiones registradas.")
            if ls["n"]:
                st.caption(f"Correcciones globales del torneo (online "
                           f"learning, {ls['n']} partidos): ritmo de goles "
                           f"×{ls['gamma']:.3f} · empates "
                           f"×{ls['draw_mult']:.3f} · altitud "
                           f"×{ls['alt_mult']:.3f}.")

# ------------------------------------------------ TAB 2: ingresar resultado
with tab_result:
    st.subheader("Registrar un resultado jugado")
    if pending.empty:
        st.info("No hay fixtures pendientes de fase de grupos.")
    else:
        opts = {f"{r.date.date()} · Grupo {r.group} · {r.home_team} vs "
                f"{r.away_team}": i for i, r in pending.iterrows()}
        pick = st.selectbox("Partido", list(opts))
        row = pending.loc[opts[pick]]
        c1, c2, c3 = st.columns([1, 1, 1])
        gh = c1.number_input(row.home_team, 0, 15, 0, key="gh")
        ga = c2.number_input(row.away_team, 0, 15, 0, key="ga")
        details = detail_block("g", row.home_team, row.away_team)
        if c3.button("Guardar resultado", type="primary"):
            save_match(row.date, row.home_team, row.away_team, gh, ga,
                       row.neutral, "group", details, prefix="g")
            st.success(f"{row.home_team} {gh} - {ga} {row.away_team} "
                       "guardado. Predicciones recalculadas.")
            st.rerun()

    st.divider()
    st.subheader("Resultado de eliminatoria / partido manual")
    teams48 = sorted(load_teams().name_canonical)
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 2, 1])
    mh = c1.selectbox("Local", teams48, key="mh")
    mgh = c2.number_input("Goles", 0, 15, 0, key="mgh")
    mga = c3.number_input("Goles ", 0, 15, 0, key="mga")
    ma = c4.selectbox("Visitante", teams48, index=1, key="ma")
    mdate = c5.date_input("Fecha", pd.Timestamp("2026-06-28"))
    winner = None
    if mgh == mga:
        wpick = st.selectbox("Ganador en prórroga/penales (eliminatoria)",
                             ["No aplica (fase de grupos)", mh, ma],
                             key="mwin")
        winner = None if wpick.startswith("No aplica") else wpick
    mdetails = detail_block("m", mh, ma)
    if st.button("Guardar partido manual"):
        if mh == ma:
            st.error("Elige dos selecciones distintas.")
        else:
            save_match(pd.Timestamp(mdate), mh, ma, mgh, mga,
                       mh not in HOSTS, "ko", mdetails, ko_winner=winner,
                       prefix="m")
            st.success("Guardado.")
            st.rerun()

    if len(live):
        st.divider()
        st.subheader(f"Resultados ingresados ({len(live)})")
        show = live[["date", "home_team", "home_score", "away_score",
                     "away_team", "xg_home", "xg_away", "weather",
                     "stage"]].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        show = show.rename(columns={
            "date": "Fecha", "home_team": "Local", "home_score": "G",
            "away_score": "G ", "away_team": "Visitante", "xg_home": "xG",
            "xg_away": "xG ", "weather": "Clima", "stage": "Fase"})
        st.markdown(tbl(show.fillna("—").sort_values("Fecha",
                                                     ascending=False),
                        flags={"Local", "Visitante"}, height=320),
                    unsafe_allow_html=True)
        if st.button("Borrar el último resultado"):
            STORE.delete_match(str(live.iloc[-1].match_id))
            st.rerun()

# ------------------------------------------------ TAB 3: líderes
with tab_leaders:
    st.subheader("Líderes del torneo")
    pl = STORE.players()
    cards_df = STORE.discipline()
    if pl.empty and cards_df.empty:
        st.info("Aún no hay eventos ingresados. Registra goleadores y "
                "asistencias al guardar cada resultado.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Goleadores**")
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
                       .rename(columns={"player": "Jugador",
                                        "team": "Selección"}))
                st.markdown(tbl(top[["Jugador", "Selección", "Goles",
                                     "Penales"]],
                                flags={"Selección"}, height=420),
                            unsafe_allow_html=True)
        with c2:
            st.markdown("**Asistencias**")
            a = pl[pl.event == "assist"]
            if a.empty:
                st.caption("Sin asistencias registradas.")
            else:
                top = (a.groupby(["player", "team"]).size()
                       .reset_index(name="Asistencias")
                       .sort_values("Asistencias", ascending=False)
                       .rename(columns={"player": "Jugador",
                                        "team": "Selección"}))
                st.markdown(tbl(top, flags={"Selección"}, height=420),
                            unsafe_allow_html=True)
        with c3:
            st.markdown("**Tarjetas**")
            if cards_df.empty:
                st.caption("Sin tarjetas registradas.")
            else:
                top = (cards_df.groupby(["player", "team"]).agg(
                        Amarillas=("card", lambda s: int((s == "yellow").sum())),
                        Rojas=("card", lambda s: int((s == "red").sum())))
                       .reset_index()
                       .sort_values(["Rojas", "Amarillas"], ascending=False)
                       .rename(columns={"player": "Jugador",
                                        "team": "Selección"}))
                st.markdown(tbl(top, flags={"Selección"}, height=420),
                            unsafe_allow_html=True)

# ------------------------------------------------ TAB 4: cuadro (bracket)
with tab_bracket:
    c1, c2 = st.columns([3, 1])
    c1.subheader("Cuadro del Mundial")
    n_sims_b = c2.selectbox("Simulaciones", [2000, 5000, 10000], index=1,
                            key="nsims_bracket")
    st.caption("Cada llave muestra el ocupante más probable según el Monte "
               "Carlo (P de llegar a ese cruce) y en rojo el favorito a "
               "avanzar. Se autocompleta con cada resultado que ingreses; "
               "los cruces ya definidos quedan al 100%.")
    with st.spinner("Simulando el torneo..."):
        simdf, slots = run_simulation(STORE.token(), n_sims_b)

    ko = sorted(load_ko_raw(), key=lambda m: m.get("num", 999))
    ko = [m for m in ko if m["round"] != "Match for third place"]
    rounds_order = ["Round of 32", "Round of 16", "Quarter-final",
                    "Semi-final", "Final"]
    round_lbl = {"Round of 32": "Dieciseisavos", "Round of 16": "Octavos",
                 "Quarter-final": "Cuartos", "Semi-final": "Semifinal",
                 "Final": "Final"}

    def side_html(cands: list, winner: str | None) -> str:
        if not cands:
            return '<div class="b-side"><span class="b-name">—</span></div>'
        team, share = cands[0]
        win = " b-win" if winner and team == winner else ""
        pct = "" if share >= 0.995 else f'<span class="b-p">{100 * share:.0f}%</span>'
        return (f'<div class="b-side{win}">{flag_img(team, 18)}'
                f'<span class="b-name">{esc(team)}</span>{pct}</div>')

    cols_html = []
    final_key = None
    for rnd in rounds_order:
        boxes = []
        for i, m in enumerate(ko):
            if m["round"] != rnd:
                continue
            key = str(m.get("num", f"x{i}"))
            ss = slots.get(key, {"t1": [], "t2": [], "w": []})
            winner = ss["w"][0][0] if ss["w"] else None
            if rnd == "Final":
                final_key = key
            fecha = pd.Timestamp(m["date"]).strftime("%d %b").upper()
            boxes.append(
                f'<div class="b-match">{side_html(ss["t1"], winner)}'
                f'{side_html(ss["t2"], winner)}'
                f'<div class="b-foot">{fecha} · {esc(m.get("ground", ""))}'
                f'</div></div>')
        cols_html.append(f'<div class="b-col"><div class="b-round">'
                         f'{round_lbl[rnd]}</div>{"".join(boxes)}</div>')

    champ_html = ""
    if final_key and slots.get(final_key, {}).get("w"):
        team, share = slots[final_key]["w"][0]
        champ_html = (
            f'<div class="b-col"><div class="b-round">Campeón</div>'
            f'<div class="b-match b-champ"><div class="b-tag">🏆 Favorito'
            f'</div><div class="b-side">{flag_img(team, 22)}'
            f'<span class="b-name">{esc(team)}</span>'
            f'<span class="b-p">{100 * share:.0f}%</span></div></div></div>')

    st.markdown(f'<div class="bracket">{"".join(cols_html)}{champ_html}</div>',
                unsafe_allow_html=True)

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
