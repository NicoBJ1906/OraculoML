"""Construcción de la zona silver (limpia, tipada, normalizada)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from mundial.config import RAW
from mundial.transform import names

log = logging.getLogger(__name__)

INTL = RAW / "international"
WC = RAW / "worldcup2026"


def load_former() -> pd.DataFrame:
    return pd.read_csv(
        INTL / "former_names.csv", parse_dates=["start_date", "end_date"]
    )


def build_matches(former: pd.DataFrame) -> pd.DataFrame:
    """Partidos jugados (con resultado), normalizados y tipados."""
    df = pd.read_csv(INTL / "results.csv", parse_dates=["date"])
    df["home_team"] = names.apply_former_names(df, former, "home_team")
    df["away_team"] = names.apply_former_names(df, former, "away_team")
    df["home_team"] = names.normalize_team(df["home_team"])
    df["away_team"] = names.normalize_team(df["away_team"])
    df["tournament"] = names.clean_text(df["tournament"])
    df["city"] = names.clean_text(df["city"])
    df["country"] = names.normalize_team(df["country"])
    df["neutral"] = df["neutral"].astype("string").str.upper().isin(["TRUE", "1", "YES"])

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype("int16")
    played["away_score"] = played["away_score"].astype("int16")
    played["year"] = played["date"].dt.year.astype("int16")
    played["total_goals"] = (played["home_score"] + played["away_score"]).astype("int16")
    played["result"] = np.select(
        [played.home_score > played.away_score, played.home_score == played.away_score],
        ["H", "D"],
        default="A",
    )
    played = played.drop_duplicates(
        subset=["date", "home_team", "away_team", "tournament"]
    )
    return played.sort_values("date").reset_index(drop=True)


def build_goalscorers(former: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(INTL / "goalscorers.csv", parse_dates=["date"])
    for col in ("home_team", "away_team", "team"):
        df[col] = names.normalize_team(names.apply_former_names(df, former, col))
    df["scorer"] = names.normalize_player(df["scorer"])
    df["scorer_ascii"] = names.strip_accents(df["scorer"])
    df["own_goal"] = df["own_goal"].fillna(False).astype(bool)
    df["penalty"] = df["penalty"].fillna(False).astype(bool)
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").astype("Int16")
    return df.sort_values("date").reset_index(drop=True)


def build_shootouts(former: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(INTL / "shootouts.csv", parse_dates=["date"])
    for col in ("home_team", "away_team", "winner", "first_shooter"):
        if col in df.columns:
            df[col] = names.normalize_team(names.apply_former_names(df, former, col))
    return df.sort_values("date").reset_index(drop=True)


def build_teams_2026() -> pd.DataFrame:
    raw = json.loads((WC / "worldcup.teams.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["name_canonical"] = names.normalize_team(df["name"])
    keep = [c for c in ("name", "name_canonical", "fifa_code", "confed", "continent", "group") if c in df.columns]
    return df[keep]


def build_stadiums_2026() -> pd.DataFrame:
    raw = json.loads((WC / "worldcup.stadiums.json").read_text(encoding="utf-8"))
    stadiums = raw.get("stadiums", raw) if isinstance(raw, dict) else raw
    df = pd.DataFrame(stadiums)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = names.clean_text(df[col])
    return df
