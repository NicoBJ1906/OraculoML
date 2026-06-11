"""Genera data/processed/rosters_2026.parquet (Gold, spec §2.4).

Plantillas normalizadas por selección a partir del histórico de goleadores
(goles desde 2022) — alimenta los dropdowns anti-typos de la UI.

Uso:
    python scripts/06_build_rosters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundial.ingest.rosters import (  # noqa: E402
    build_rosters, build_rosters_tm, citizens_by_value, merge_rosters,
)
from mundial.transform.names import TM_COUNTRY_ALIASES  # noqa: E402

TM = ROOT / "data/raw/transfermarkt"


def main() -> None:
    gs = pd.read_csv(ROOT / "data/raw/international/goalscorers.csv")
    teams = pd.read_parquet(
        ROOT / "data/interim/teams_2026.parquet")["name_canonical"].tolist()
    out = build_rosters(gs, teams)

    # Transfermarkt (si está descargado): plantel ACTIVO completo, no solo
    # goleadores — corrige dropdowns con jugadores faltantes
    if (TM / "player_national_performances.csv").exists():
        perf = pd.read_csv(TM / "player_national_performances.csv",
                           usecols=["player_id", "goals", "career_state"])
        profiles = pd.read_csv(TM / "player_profiles.csv",
                               usecols=["player_id", "player_name",
                                        "citizenship"])
        tm = build_rosters_tm(perf, profiles, teams, TM_COUNTRY_ALIASES)
        values = pd.read_csv(TM / "player_market_value.csv",
                             parse_dates=["date_unix"]).rename(
            columns={"date_unix": "date"})
        cit = citizens_by_value(profiles, values, teams, TM_COUNTRY_ALIASES)
        print(f"Transfermarkt: {len(tm):,} internacionales activos + "
              f"{len(cit):,} ciudadanos con valoración vigente")
        out = merge_rosters(tm, out, cit)

    dest = ROOT / "data/processed/rosters_2026.parquet"
    out.to_parquet(dest, index=False)
    print(f"OK: {len(out):,} jugadores de {out.team.nunique()} selecciones "
          f"-> {dest}")


if __name__ == "__main__":
    main()
