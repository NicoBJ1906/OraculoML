"""Tests de los fixes del modelo (sesión 2026-06-10):
K ponderado por torneo, tiebreak comprimido, rest_days capado e
incertidumbre de fuerza (elo_sigma) en el Monte Carlo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mundial.features.build import REST_DAYS_CAP, build_features
from mundial.features.elo import BASE, K, compute_elo, k_for
from mundial.predict.engine import tiebreak_prob

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# k_for
# ---------------------------------------------------------------------------

def test_k_por_torneo():
    assert k_for("FIFA World Cup") == 60.0
    assert k_for("Copa América") == 50.0
    assert k_for("UEFA Euro") == 50.0
    assert k_for("Gold Cup") == 50.0
    assert k_for("FIFA World Cup qualification") == 40.0
    assert k_for("UEFA Nations League") == 40.0
    assert k_for("Gold Cup qualification") == 40.0     # qualification gana
    assert k_for("Friendly") == 20.0
    assert k_for("Merdeka Tournament") == K            # menor -> default
    assert k_for(None) == K
    assert k_for(float("nan")) == K                    # NaN de pandas


# ---------------------------------------------------------------------------
# tiebreak_prob
# ---------------------------------------------------------------------------

def test_tiebreak_simetrico_y_comprimido():
    assert tiebreak_prob(1500, 1500) == pytest.approx(0.5)
    # complementariedad
    assert tiebreak_prob(2100, 1900) + tiebreak_prob(1900, 2100) == \
        pytest.approx(1.0)
    # comprimido: incluso con +400 Elo los penales no pasan de ~0.62
    assert 0.55 < tiebreak_prob(1900, 1500) < 0.62
    assert 0.38 < tiebreak_prob(1500, 1900) < 0.45


# ---------------------------------------------------------------------------
# compute_elo con K ponderado
# ---------------------------------------------------------------------------

def _mini_matches(tournament: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [1], "away_score": [0],
        "neutral": [True], "tournament": [tournament],
    })


def test_compute_elo_pondera_por_torneo():
    """Una victoria en Mundial mueve el rating 3x más que un amistoso."""
    base = compute_elo(_mini_matches("Friendly").drop(columns="tournament"))
    assert base.elo_home_pre.iloc[0] == BASE       # sin columna no explota

    # el delta es proporcional a K: lo medimos con un 2.º partido ficticio
    def delta(t: str) -> float:
        m = pd.concat([_mini_matches(t), _mini_matches(t)],
                      ignore_index=True)
        m.loc[1, "date"] = pd.Timestamp("2024-02-01")
        out = compute_elo(m)
        return out.elo_home_pre.iloc[1] - BASE

    assert delta("FIFA World Cup") == pytest.approx(3 * delta("Friendly"))


# ---------------------------------------------------------------------------
# rest_days capado (paridad build <-> engine)
# ---------------------------------------------------------------------------

def test_rest_days_capado_en_build():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [1, 1], "away_score": [0, 0],
        "neutral": [True, True], "tournament": ["Friendly", "Friendly"],
    })
    out = build_features(m)
    assert out.home_rest_days.iloc[1] == REST_DAYS_CAP   # 365 -> 30


def test_rest_days_capado_en_engine():
    from mundial.predict.engine import PredictionEngine

    eng = PredictionEngine.__new__(PredictionEngine)   # sin modelos: solo features

    eng.elo, eng.form, eng.h2h = {}, {}, {}
    eng.last_date = {"A": pd.Timestamp("2023-01-01")}
    eng.n_matches = {}
    eng._logval, eng._logval_max = {}, 0
    X = eng.features_for("2024-01-01", "A", "B")
    assert X.home_rest_days.iloc[0] == REST_DAYS_CAP


# ---------------------------------------------------------------------------
# Valor de plantilla (Transfermarkt) y ensemble de 3 modelos
# ---------------------------------------------------------------------------

def _sv() -> pd.DataFrame:
    return pd.DataFrame({"team": ["A", "A", "B"], "year": [2024, 2025, 2025],
                         "log_value": [8.0, 8.5, 7.5]})


def test_merge_squad_values_en_build():
    m = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [1], "away_score": [0],
        "neutral": [True], "tournament": ["Friendly"], "year": [2025],
    })
    out = build_features(m, squad_values=_sv())
    assert out.home_log_value.iloc[0] == 8.5
    assert out.diff_log_value.iloc[0] == pytest.approx(1.0)


def test_engine_log_value_con_fallback():
    from mundial.predict.engine import PredictionEngine

    eng = PredictionEngine.__new__(PredictionEngine)
    eng._logval = {("A", 2024): (8.0, 27.0, 0.3), ("A", 2025): (8.5, 26.0, 0.4)}
    eng._logval_max = 2025
    assert eng._log_value("A", 2025)[0] == 8.5
    assert eng._log_value("A", 2027)[0] == 8.5    # cap al último snapshot
    assert eng._log_value("A", 2026)[1] == 26.0   # fallback 1 año atrás
    assert pd.isna(eng._log_value("B", 2025)[0])  # sin cobertura


def test_mix_tres_modelos_y_xgb_inactivo():
    import numpy as np

    from mundial.predict.engine import PredictionEngine

    eng = PredictionEngine.__new__(PredictionEngine)
    p1 = np.array([0.2, 0.3, 0.5])
    p2 = np.array([0.4, 0.3, 0.3])
    p3 = np.array([0.3, 0.4, 0.3])

    eng.xgb, eng.weights = object(), (0.5, 0.3, 0.2)
    eng.xgb_active = True
    out = eng._mix(p1, p2, p3)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5 * 0.2 + 0.3 * 0.4 + 0.2 * 0.3)

    # peso 0 -> xgb inactivo: renormaliza clf+pois y no necesita p_xgb
    eng.weights, eng.xgb_active = (0.7, 0.0, 0.3), False
    out = eng._mix(p1, None, p3)
    assert out.sum() == pytest.approx(1.0)
    assert out[2] == pytest.approx((0.7 * 0.5 + 0.3 * 0.3) / 1.0)

    # back-compat: sin weights usa blend
    eng.weights, eng.blend = None, 0.8
    out = eng._mix(p1, None, p3)
    assert out[0] == pytest.approx(0.8 * 0.2 + 0.2 * 0.3)


# ---------------------------------------------------------------------------
# Monte Carlo: elo_sigma (integración con artefactos reales)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_factory():
    import joblib

    from mundial.predict.engine import PredictionEngine
    from mundial.predict.montecarlo import TournamentSimulator

    art = joblib.load(ROOT / "models" / "artifacts.joblib")
    matches = pd.read_parquet(ROOT / "data" / "interim" / "matches.parquet")
    eng = PredictionEngine(matches, art["clf"], art["pois_home"],
                           art["pois_away"], art["rho"], art["blend"])

    teams = pd.read_parquet(ROOT / "data" / "interim" / "teams_2026.parquet")
    groups = teams.groupby("group")["name_canonical"].apply(list).to_dict()
    df = pd.read_csv(ROOT / "data" / "raw" / "international" / "results.csv",
                     parse_dates=["date"])
    fx = df[df.home_score.isna() & (df.tournament == "FIFA World Cup")
            & (df.date.dt.year == 2026)].copy()
    fx["neutral"] = fx["neutral"].astype("string").str.upper().isin(
        ["TRUE", "1"])
    fx["group"] = fx["home_team"].map(
        teams.set_index("name_canonical")["group"])
    fx = fx[["date", "home_team", "away_team", "city", "country",
             "neutral", "group"]].sort_values("date").reset_index(drop=True)
    ko = [x for x in json.loads(
        (ROOT / "data" / "raw" / "worldcup2026" / "worldcup.json")
        .read_text(encoding="utf-8"))["matches"] if "group" not in x]
    live = pd.DataFrame(columns=["home_team", "away_team",
                                 "home_score", "away_score"])

    def make():
        return eng, TournamentSimulator(eng, fx, live, ko, groups)

    return make


def test_mc_sigma_cero_es_determinista(sim_factory):
    _, sim = sim_factory()
    a = sim.run(60, seed=11, elo_sigma=0.0)
    b = sim.run(60, seed=11, elo_sigma=0.0)
    pd.testing.assert_frame_equal(a, b)


def test_mc_sigma_restaura_elo_y_normaliza(sim_factory):
    eng, sim = sim_factory()
    before = dict(eng.elo)
    res = sim.run(60, seed=7, elo_sigma=60.0, block=30)
    assert eng.elo == before                       # restaurado exacto
    assert res.CAMPEON.sum() == pytest.approx(1.0)
    assert ((res.R32 >= 0) & (res.R32 <= 1)).all()
