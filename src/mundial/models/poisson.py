"""Modelo de goles Poisson (con corrección Dixon-Coles para marcadores bajos).

Dos regresiones Poisson independientes (goles local / goles visitante) sobre
features pre-partido. De las lambdas se construye la matriz de marcadores y de
ahí P(1), P(X), P(2), over/under y el marcador más probable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POISSON_FEATURES = [
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "home_form_gf", "home_form_ga",
    "away_form_gf", "away_form_ga",
    "neutral",
]

MAX_GOALS = 10  # la matriz cubre 0..MAX_GOALS goles por equipo


def make_poisson() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", PoissonRegressor(alpha=1e-4, max_iter=2000)),
    ])


def fit_goal_models(X: pd.DataFrame, home_goals, away_goals) -> tuple[Pipeline, Pipeline]:
    """Entrena el modelo de goles del local y del visitante."""
    mh = make_poisson().fit(X, home_goals)
    ma = make_poisson().fit(X, away_goals)
    return mh, ma


def _tau(matrix: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    """Corrección Dixon-Coles: ajusta la dependencia en marcadores 0-0/1-0/0-1/1-1."""
    m = matrix.copy()
    m[0, 0] *= max(1.0 - lh * la * rho, 0.0)
    m[0, 1] *= max(1.0 + lh * rho, 0.0)
    m[1, 0] *= max(1.0 + la * rho, 0.0)
    m[1, 1] *= max(1.0 - rho, 0.0)
    return m / m.sum()


def score_matrix(lh: float, la: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matriz P(home=i, away=j) para i,j en 0..max_goals."""
    from scipy.stats import poisson

    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, lh)
    pa = poisson.pmf(i, la)
    m = np.outer(ph, pa)
    if rho:
        m = _tau(m, lh, la, rho)
    return m / m.sum()


def outcome_probs(matrix: np.ndarray) -> dict[str, float]:
    """P(H), P(D), P(A) desde la matriz de marcadores."""
    return {
        "H": float(np.tril(matrix, -1).sum()),   # home > away
        "D": float(np.trace(matrix)),
        "A": float(np.triu(matrix, 1).sum()),    # away > home
    }


def top_scorelines(matrix: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    """Los n marcadores más probables como [('2-1', 0.081), ...]."""
    flat = [(f"{i}-{j}", float(matrix[i, j]))
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    return sorted(flat, key=lambda x: -x[1])[:n]


def predict_proba_1x2(model_home: Pipeline, model_away: Pipeline,
                      X: pd.DataFrame, rho: float = 0.0) -> np.ndarray:
    """Probabilidades 1X2 derivadas del Poisson, en orden de clases ['A','D','H']
    (el mismo orden alfabético que usa sklearn en el clasificador)."""
    lh = model_home.predict(X)
    la = model_away.predict(X)
    out = np.empty((len(X), 3))
    for k in range(len(X)):
        p = outcome_probs(score_matrix(lh[k], la[k], rho))
        out[k] = [p["A"], p["D"], p["H"]]
    return out


def estimate_rho(model_home: Pipeline, model_away: Pipeline,
                 X_val: pd.DataFrame, y_val, grid=None) -> float:
    """Busca el rho de Dixon-Coles que minimiza el log-loss en validación."""
    from sklearn.metrics import log_loss

    if grid is None:
        grid = np.linspace(-0.30, 0.15, 19)
    best_rho, best_ll = 0.0, np.inf
    for rho in grid:
        proba = predict_proba_1x2(model_home, model_away, X_val, rho)
        ll = log_loss(y_val, proba, labels=["A", "D", "H"])
        if ll < best_ll:
            best_rho, best_ll = float(rho), ll
    return best_rho
