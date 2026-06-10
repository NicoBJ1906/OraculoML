"""Motor de predicción en vivo para el Mundial 2026.

Mantiene el estado de cada selección (Elo, forma, H2H) construido desde el
histórico completo y lo actualiza con cada resultado nuevo que se ingresa
durante el torneo. Replica EXACTAMENTE la construcción de features de
`mundial.features.build` para que el modelo entrenado reciba lo mismo que vio
en entrenamiento.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from mundial.features.elo import BASE, HFA, K, _g_multiplier
from mundial.models.baseline import FEATURES
from mundial.models.poisson import (
    POISSON_FEATURES, outcome_probs, score_matrix, top_scorelines,
)

FORM_WINDOW = 5


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


class PredictionEngine:
    """Estado incremental + predicción con clasificador y Poisson."""

    def __init__(self, matches: pd.DataFrame, clf, pois_home, pois_away,
                 rho: float = 0.0, blend: float = 0.5):
        """matches: silver matches (histórico completo, ordenado o no)."""
        self.clf = clf
        self.pois_home = pois_home
        self.pois_away = pois_away
        self.rho = rho
        self.blend = blend  # peso del clasificador en el ensemble

        self.elo: dict[str, float] = {}
        self.form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
        self.last_date: dict[str, pd.Timestamp] = {}
        self.n_matches: dict[str, int] = defaultdict(int)
        self.h2h: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])

        m = matches.sort_values("date", kind="stable")
        for row in m.itertuples(index=False):
            self.apply_result(row.date, row.home_team, row.away_team,
                              int(row.home_score), int(row.away_score),
                              bool(row.neutral))

    # ------------------------------------------------ estado
    def apply_result(self, date, home: str, away: str,
                     hs: int, as_: int, neutral: bool = True) -> None:
        """Actualiza Elo, forma, H2H y fechas con un partido jugado."""
        date = pd.Timestamp(date)
        rh = self.elo.get(home, BASE)
        ra = self.elo.get(away, BASE)
        adv = 0.0 if neutral else HFA
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - adv) / 400.0))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = K * _g_multiplier(hs - as_) * (s_h - exp_h)
        self.elo[home] = rh + delta
        self.elo[away] = ra - delta

        pts_h = 3 if hs > as_ else (1 if hs == as_ else 0)
        pts_a = 3 if as_ > hs else (1 if hs == as_ else 0)
        self.form[home].append((pts_h, hs, as_))
        self.form[away].append((pts_a, as_, hs))
        self.last_date[home] = date
        self.last_date[away] = date
        self.n_matches[home] += 1
        self.n_matches[away] += 1

        x, y = _pair(home, away)
        c = self.h2h[(x, y)]
        if hs > as_:
            c[0 if home == x else 2] += 1
        elif hs < as_:
            c[2 if home == x else 0] += 1
        else:
            c[1] += 1

    # ------------------------------------------------ hooks (capa live)
    # Subclases (LiveEngine) los sobreescriben; aquí son identidad para que
    # el engine base se comporte EXACTAMENTE igual que en entrenamiento.
    def elo_for(self, team: str, date=None) -> float:
        """Elo efectivo para predecir (la capa live suma ajustes aquí)."""
        return self.elo.get(team, BASE)

    def _adjust_lambdas(self, date, home: str, away: str, lh: float,
                        la: float, city: str | None = None) -> tuple[float, float]:
        return lh, la

    def _adjust_probs(self, date, p: np.ndarray) -> np.ndarray:
        """p en orden ['A','D','H']."""
        return p

    # ------------------------------------------------ features
    def _team_form(self, team: str) -> tuple[float, float, float]:
        f = self.form.get(team)
        if not f:
            return (np.nan, np.nan, np.nan)
        arr = np.array(f, dtype=float)
        return tuple(arr.mean(axis=0))  # (pts, gf, ga) medios últimos <=5

    def features_for(self, date, home: str, away: str,
                     neutral: bool = True) -> pd.DataFrame:
        """Fila de features pre-partido con el estado actual (sin leakage)."""
        date = pd.Timestamp(date)
        hp, hgf, hga = self._team_form(home)
        ap, agf, aga = self._team_form(away)
        eh = self.elo_for(home, date)
        ea = self.elo_for(away, date)
        x, y = _pair(home, away)
        c = self.h2h.get((x, y), [0, 0, 0])
        h2h_h, h2h_d, h2h_a = (c[0], c[1], c[2]) if home == x else (c[2], c[1], c[0])
        rest_h = (date - self.last_date[home]).days if home in self.last_date else np.nan
        rest_a = (date - self.last_date[away]).days if away in self.last_date else np.nan

        row = {
            "elo_home_pre": eh, "elo_away_pre": ea, "elo_diff": eh - ea,
            "home_form_pts": hp, "home_form_gf": hgf, "home_form_ga": hga,
            "away_form_pts": ap, "away_form_gf": agf, "away_form_ga": aga,
            "diff_form_pts": hp - ap, "diff_form_gf": hgf - agf,
            "diff_form_ga": hga - aga,
            "h2h_home_wins": float(h2h_h), "h2h_draws": float(h2h_d),
            "h2h_away_wins": float(h2h_a),
            "home_rest_days": rest_h, "away_rest_days": rest_a,
            "diff_rest_days": rest_h - rest_a,
            "neutral": int(neutral),
        }
        return pd.DataFrame([row])

    # ------------------------------------------------ predicción
    def match_distribution(self, date, home: str, away: str,
                           neutral: bool = True,
                           city: str | None = None) -> dict:
        """Distribución completa de un partido: P(1X2) ensemble + matriz de
        marcadores reescalada para ser coherente con esas probabilidades.
        Es la base de predict_match y de la simulación Monte Carlo."""
        X = self.features_for(date, home, away, neutral)
        p_clf = self.clf.predict_proba(X[FEATURES])[0]          # orden A, D, H
        lh = float(self.pois_home.predict(X[POISSON_FEATURES])[0])
        la = float(self.pois_away.predict(X[POISSON_FEATURES])[0])
        lh, la = self._adjust_lambdas(date, home, away, lh, la, city)
        matrix = score_matrix(lh, la, self.rho)
        pp = outcome_probs(matrix)
        p_pois = np.array([pp["A"], pp["D"], pp["H"]])
        p = self.blend * p_clf + (1 - self.blend) * p_pois
        p = self._adjust_probs(date, p)

        order = list(self.clf.classes_)  # ['A','D','H']
        probs = dict(zip(order, p))

        # Reescala la matriz de marcadores para que sus marginales 1X2
        # coincidan con el ensemble (coherencia entre marcador y porcentajes).
        n = matrix.shape[0]
        tri_h = np.tril(np.ones((n, n), dtype=bool), -1)   # home > away
        tri_d = np.eye(n, dtype=bool)
        tri_a = np.triu(np.ones((n, n), dtype=bool), 1)
        matrix = matrix.copy()
        for region, lbl in ((tri_h, "H"), (tri_d, "D"), (tri_a, "A")):
            mass = matrix[region].sum()
            if mass > 0:
                matrix[region] *= probs[lbl] / mass
        matrix /= matrix.sum()

        # P(ganar prórroga/penales) si hay empate en 90' ~ Elo
        p_tiebreak = 1.0 / (1.0 + 10 ** ((self.elo_for(away, date)
                                          - self.elo_for(home, date)) / 400.0))
        return {"probs": probs, "matrix": matrix, "lambda_home": lh,
                "lambda_away": la, "p_home_tiebreak": p_tiebreak,
                "tri": (tri_h, tri_d, tri_a)}

    def predict_match(self, date, home: str, away: str,
                      neutral: bool = True, city: str | None = None) -> dict:
        """Predicción completa: P(1X2) ensemble, lambdas y marcadores top."""
        d = self.match_distribution(date, home, away, neutral, city)
        probs, matrix = d["probs"], d["matrix"]
        lh, la = d["lambda_home"], d["lambda_away"]
        tri_h, tri_d, tri_a = d["tri"]
        pred = max(probs, key=probs.get)

        # Marcador más probable DENTRO del resultado predicho (para la card).
        region_pred = {"H": tri_h, "D": tri_d, "A": tri_a}[pred]
        masked = np.where(region_pred, matrix, 0.0)
        i, j = np.unravel_index(int(masked.argmax()), masked.shape)
        score_pred = (f"{i}-{j}", float(matrix[i, j]))
        p_win_tiebreak = d["p_home_tiebreak"]
        return {
            "home": home, "away": away,
            "p_home": float(probs["H"]), "p_draw": float(probs["D"]),
            "p_away": float(probs["A"]),
            "pred": {"H": home, "D": "Empate", "A": away}[pred],
            "lambda_home": lh, "lambda_away": la,
            "scorelines": top_scorelines(matrix, 5),
            "score_pred": score_pred,
            "elo_home": self.elo_for(home, date),
            "elo_away": self.elo_for(away, date),
            "p_home_advances": float(probs["H"] + probs["D"] * p_win_tiebreak),
            "p_away_advances": float(probs["A"] + probs["D"] * (1 - p_win_tiebreak)),
        }

    def elo_ranking(self, teams: list[str] | None = None) -> pd.DataFrame:
        elo = self.elo if teams is None else {t: self.elo.get(t, BASE) for t in teams}
        df = pd.DataFrame(sorted(elo.items(), key=lambda x: -x[1]),
                          columns=["team", "elo"])
        df.index = np.arange(1, len(df) + 1)
        return df
