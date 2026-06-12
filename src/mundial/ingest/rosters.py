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


def build_rosters_tm(perf: pd.DataFrame, profiles: pd.DataFrame,
                     teams: list[str],
                     aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Plantillas desde Transfermarkt: internacionales ACTIVOS por selección
    (career_state CURRENT/RECENT), con sus goles de selección. Cubre a todo
    el plantel, no solo a quienes ya marcaron (limitación de goalscorers).

    Args:
        perf: player_national_performances (player_id, goals, career_state).
        profiles: player_profiles (player_id, player_name, citizenship).
        teams: nombres canónicos de las selecciones clasificadas.
        aliases: ciudadanía Transfermarkt -> nombre canónico.

    Returns:
        DataFrame [team, player, goals, last_seen] (last_seen vacío: la
        fuente no trae fecha del último partido).
    """
    act = perf[perf["career_state"].isin(
        ["CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"])]
    pp = profiles[["player_id", "player_name", "citizenship"]].copy()
    # ciudadanía múltiple separada por DOBLE espacio; la 1.ª es la relevante
    pp["team"] = (pp["citizenship"].astype("string")
                  .str.split(r"  +", regex=True).str[0].str.strip())
    if aliases:
        pp["team"] = pp["team"].replace(aliases)
    # el nombre viene como "Jugador (id)" -> limpiar el sufijo
    pp["player"] = (pp["player_name"].astype("string")
                    .str.replace(r"\s*\(\d+\)$", "", regex=True).str.strip())

    j = act.merge(pp[["player_id", "team", "player"]], on="player_id")
    j = j[j["team"].isin(set(teams)) & j["player"].notna()]
    out = (j.groupby(["team", "player"])
            .agg(goals=("goals", "max"))
            .reset_index())
    out["last_seen"] = pd.NaT
    return out.sort_values(["team", "goals", "player"],
                           ascending=[True, False, True]).reset_index(drop=True)


def citizens_by_value(profiles: pd.DataFrame, values: pd.DataFrame,
                      teams: list[str], aliases: dict[str, str] | None = None,
                      top_n: int = 120, since: str = "2024-07-01") -> pd.DataFrame:
    """Capa de respaldo: ciudadanos con valoración de mercado reciente,
    top_n por valor por selección. Cubre naturalizados y debutantes que aún
    no figuran en player_national_performances (ej. Julián Quiñones con
    México). goals=0 (van al final del dropdown)."""
    pp = profiles[["player_id", "player_name", "citizenship"]].copy()
    pp["team"] = (pp["citizenship"].astype("string")
                  .str.split(r"  +", regex=True).str[0].str.strip())
    if aliases:
        pp["team"] = pp["team"].replace(aliases)
    pp = pp[pp["team"].isin(set(teams))]
    pp["player"] = (pp["player_name"].astype("string")
                    .str.replace(r"\s*\(\d+\)$", "", regex=True).str.strip())

    v = values[values["date"] >= pd.Timestamp(since)]
    last = v.groupby("player_id")["value"].last().rename("mv")
    j = pp.merge(last, on="player_id").dropna(subset=["mv"])
    j = (j.sort_values("mv", ascending=False).groupby("team").head(top_n))
    out = j[["team", "player"]].copy()
    out["goals"] = 0
    out["last_seen"] = pd.NaT
    return out


def merge_rosters(*sources: pd.DataFrame) -> pd.DataFrame:
    """Une fuentes sin duplicar (team, player); la primera fuente manda.
    El dedupe ignora acentos ("Katić" == "Katic" entre fuentes)."""
    from mundial.transform.names import strip_accents

    both = pd.concat(sources, ignore_index=True)
    key = strip_accents(both["player"].astype("string")).str.lower()
    out = both[~pd.DataFrame({"team": both["team"], "k": key}).duplicated()]
    return out.sort_values(["team", "goals", "player"],
                           ascending=[True, False, True]).reset_index(drop=True)
