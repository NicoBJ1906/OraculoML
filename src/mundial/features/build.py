"""Feature engineering (zona gold). Todas las features miran SOLO al pasado.

Salida: una fila por partido con features de local y visitante calculadas con
información previa a la fecha del partido (sin data leakage).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from mundial.features.elo import compute_elo

FORM_FEATURES = ["form_pts", "form_gf", "form_ga", "rest_days", "matches_prior"]


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (partido, equipo) con perspectiva propia."""
    home = matches[["match_id", "date", "home_team", "away_team",
                    "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent",
                 "home_score": "gf", "away_score": "ga"})
    home["is_home"] = True
    away = matches[["match_id", "date", "away_team", "home_team",
                    "away_score", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent",
                 "away_score": "gf", "home_score": "ga"})
    away["is_home"] = False
    lf = pd.concat([home, away], ignore_index=True)
    lf["points"] = np.select([lf.gf > lf.ga, lf.gf == lf.ga], [3, 1], default=0)
    return lf.sort_values(["team", "date"]).reset_index(drop=True)


def _rolling_form(lf: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Medias móviles de los últimos `window` partidos, excluyendo el actual."""
    grp = lf.groupby("team", sort=False)
    lf["form_pts"] = grp["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    lf["form_gf"] = grp["gf"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    lf["form_ga"] = grp["ga"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    lf["rest_days"] = grp["date"].diff().dt.days
    lf["matches_prior"] = grp.cumcount()
    return lf


def _h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head histórico previo (victorias local/empates/visitante)."""
    home = matches["home_team"].to_numpy()
    away = matches["away_team"].to_numpy()
    hs = matches["home_score"].to_numpy()
    as_ = matches["away_score"].to_numpy()
    n = len(matches)
    hw = np.zeros(n)
    dd = np.zeros(n)
    aw = np.zeros(n)
    counts: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])  # (x,y) x<y -> [x_wins, draws, y_wins]

    for i in range(n):
        h, a = home[i], away[i]
        x, y = (h, a) if h < a else (a, h)
        c = counts[(x, y)]
        if h == x:
            hw[i], dd[i], aw[i] = c[0], c[1], c[2]
        else:
            hw[i], dd[i], aw[i] = c[2], c[1], c[0]
        if hs[i] > as_[i]:
            c[0 if h == x else 2] += 1
        elif hs[i] < as_[i]:
            c[2 if h == x else 0] += 1
        else:
            c[1] += 1

    matches = matches.copy()
    matches["h2h_home_wins"] = hw
    matches["h2h_draws"] = dd
    matches["h2h_away_wins"] = aw
    return matches


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla gold de features por partido."""
    m = compute_elo(matches)            # ordena por fecha + elo_*_pre
    m = _h2h(m).reset_index(drop=True)
    m["match_id"] = np.arange(len(m))

    lf = _rolling_form(_long_format(m))
    home = lf[lf.is_home].set_index("match_id")[FORM_FEATURES].add_prefix("home_")
    away = lf[~lf.is_home].set_index("match_id")[FORM_FEATURES].add_prefix("away_")

    out = m.set_index("match_id").join(home).join(away)
    for f in FORM_FEATURES:
        out[f"diff_{f}"] = out[f"home_{f}"] - out[f"away_{f}"]
    return out.reset_index()
