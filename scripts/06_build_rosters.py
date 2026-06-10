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

from mundial.ingest.rosters import build_rosters  # noqa: E402


def main() -> None:
    gs = pd.read_csv(ROOT / "data/raw/international/goalscorers.csv")
    teams = pd.read_parquet(
        ROOT / "data/interim/teams_2026.parquet")["name_canonical"].tolist()
    out = build_rosters(gs, teams)
    dest = ROOT / "data/processed/rosters_2026.parquet"
    out.to_parquet(dest, index=False)
    print(f"OK: {len(out):,} jugadores de {out.team.nunique()} selecciones "
          f"-> {dest}")


if __name__ == "__main__":
    main()
