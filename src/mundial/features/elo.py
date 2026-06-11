"""Cálculo de ratings Elo para selecciones (estilo World Football Elo).

El rating PRE-partido es un feature sin leakage: se calcula con los partidos
anteriores y se actualiza DESPUÉS de cada encuentro, en orden cronológico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = 1500.0   # rating inicial de una selección nueva
HFA = 65.0      # ventaja de local en puntos Elo
K = 30.0        # factor por defecto (torneos menores / sin info de torneo)

# K ponderado por importancia del partido (estándar World Football Elo):
# un amistoso no puede mover el rating igual que una semifinal de Mundial.
_K_CONTINENTAL = ("Copa América", "UEFA Euro", "AFC Asian Cup",
                  "African Cup of Nations", "Gold Cup",
                  "CONCACAF Championship")


def k_for(tournament: str | None) -> float:
    """Factor K según el torneo del partido."""
    if not tournament or not isinstance(tournament, str):
        return K
    if tournament == "FIFA World Cup":
        return 60.0
    if "qualification" in tournament or "Nations League" in tournament \
            or "Confederations Cup" in tournament:
        return 40.0
    if tournament in _K_CONTINENTAL:
        return 50.0
    if tournament == "Friendly":
        return 20.0
    return K


def _g_multiplier(goal_diff: int) -> float:
    """Multiplicador por margen de victoria (World Football Elo)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Añade elo_home_pre, elo_away_pre y elo_diff (valores ANTES del partido)."""
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()
    neutral = df["neutral"].to_numpy()
    if "tournament" in df.columns:
        ks = np.array([k_for(t) for t in df["tournament"]])
    else:
        ks = np.full(len(df), K)
    n = len(df)
    home_pre = np.empty(n)
    away_pre = np.empty(n)
    ratings: dict[str, float] = {}

    for i in range(n):
        h, a = home[i], away[i]
        rh = ratings.get(h, BASE)
        ra = ratings.get(a, BASE)
        home_pre[i] = rh
        away_pre[i] = ra

        adv = 0.0 if neutral[i] else HFA
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - adv) / 400.0))
        if hs[i] > as_[i]:
            s_h = 1.0
        elif hs[i] == as_[i]:
            s_h = 0.5
        else:
            s_h = 0.0
        delta = ks[i] * _g_multiplier(int(hs[i] - as_[i])) * (s_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    df["elo_home_pre"] = home_pre
    df["elo_away_pre"] = away_pre
    df["elo_diff"] = home_pre - away_pre
    return df
