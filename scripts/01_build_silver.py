"""Sprint 1 - Silver: limpieza y normalización del data lake local.

raw/ (CSV/JSON crudo)  ->  interim/ (Parquet limpio, tipado, normalizado)

Uso:
    python scripts/01_build_silver.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mundial.config import INTERIM, SETTINGS  # noqa: E402
from mundial.transform import silver  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("silver")


def save(df: pd.DataFrame, name: str) -> None:
    out = INTERIM / f"{name}.parquet"
    df.to_parquet(out, index=False)
    log.info("%-14s -> %s  (%d filas, %d cols)", name, out.name, len(df), df.shape[1])


def main() -> None:
    former = silver.load_former()
    matches = silver.build_matches(former)
    goals = silver.build_goalscorers(former)
    shootouts = silver.build_shootouts(former)
    teams = silver.build_teams_2026()
    stadiums = silver.build_stadiums_2026()

    save(matches, "matches")
    save(goals, "goalscorers")
    save(shootouts, "shootouts")
    save(teams, "teams_2026")
    save(stadiums, "stadiums_2026")

    # ----------------- validaciones de calidad -----------------
    print("\n== VALIDACIONES ==")
    print("matches:", matches.shape, "| rango:",
          matches.date.min().date(), "->", matches.date.max().date())

    key = ["home_team", "away_team", "home_score", "away_score", "result"]
    assert matches[key].notna().all().all(), "Hay NaN en columnas clave de matches"
    print("sin NaN en columnas clave: OK")

    start = SETTINGS["project"]["model_start_year"]
    m = matches[matches.year >= start]
    cov = {t: int(((m.home_team == t) | (m.away_team == t)).sum())
           for t in teams.name_canonical}
    faltan = [t for t, c in cov.items() if c == 0]
    print(f"equipos del Mundial 2026 sin partidos >= {start}:",
          faltan if faltan else "NINGUNO (cobertura 100% de los 48)")
    print("los 5 con menos partidos:",
          sorted(cov.items(), key=lambda x: x[1])[:5])

    print("distribucion resultado (H/D/A):",
          matches.result.value_counts(normalize=True).round(3).to_dict())
    print("jugadores unicos (scorer):", int(goals.scorer.nunique()))
    print("columnas stadiums_2026:", list(stadiums.columns))


if __name__ == "__main__":
    main()
