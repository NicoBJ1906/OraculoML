"""Plantillas normalizadas de jugadores (Gold, spec §2.4).

Sin una API oficial de rosters, la fuente más fiable que ya tenemos es el
histórico de goleadores (martj42): jugadores que marcaron para cada
selección recientemente. Sirve como diccionario anti-typos para los
dropdowns de la UI; el caso "jugador sin goles previos" se cubre con la
opción de texto libre "Otro…".
"""
from __future__ import annotations

import pandas as pd

DEFAULT_SINCE = "2022-06-01"


def build_rosters(goalscorers: pd.DataFrame, teams: list[str],
                  since: str = DEFAULT_SINCE) -> pd.DataFrame:
    """Construye la tabla Gold de plantillas.

    Args:
        goalscorers: histórico crudo (date, team, scorer, own_goal, ...).
        teams: nombres canónicos de las selecciones clasificadas.
        since: solo se consideran goles desde esta fecha.

    Returns:
        DataFrame [team, player, goals, last_seen] ordenado por equipo y
        goles descendentes (el orden del dropdown).
    """
    df = goalscorers.copy()
    df["date"] = pd.to_datetime(df["date"])
    own = df.get("own_goal")
    if own is not None:                      # un autogol no es "su" jugador
        df = df[~own.astype("string").str.upper().isin(["TRUE", "1"])]
    df = df[(df["date"] >= pd.Timestamp(since))
            & df["team"].isin(set(teams)) & df["scorer"].notna()]
    out = (df.groupby(["team", "scorer"])
             .agg(goals=("date", "size"), last_seen=("date", "max"))
             .reset_index()
             .rename(columns={"scorer": "player"})
             .sort_values(["team", "goals", "player"],
                          ascending=[True, False, True])
             .reset_index(drop=True))
    out["last_seen"] = out["last_seen"].dt.date
    return out
