"""Sprint 1 - Gold: feature engineering sin leakage.

interim/matches.parquet  ->  processed/features.parquet

Uso:
    python scripts/02_build_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mundial.config import INTERIM, PROCESSED  # noqa: E402
from mundial.features.build import build_features  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gold")


def main() -> None:
    matches = pd.read_parquet(INTERIM / "matches.parquet")
    log.info("matches cargados: %d", len(matches))

    feats = build_features(matches)
    out = PROCESSED / "features.parquet"
    feats.to_parquet(out, index=False)
    log.info("features -> %s  (%d filas, %d cols)", out.name, len(feats), feats.shape[1])

    # ----------------- chequeos anti-leakage -----------------
    print("\n== ANTI-LEAKAGE / SANITY ==")
    first = feats.sort_values("date").groupby("home_team").head(1)
    # el primer partido como local de un equipo no debería tener forma previa
    print("Elo del primerísimo partido (debe ser 1500):",
          round(float(feats.sort_values("date").iloc[0].elo_home_pre), 1))
    print("home_form_pts NaN en debut local (esperado, sin historial):",
          int(feats[feats.home_matches_prior == 0].home_form_pts.isna().sum()),
          "/", int((feats.home_matches_prior == 0).sum()))

    m = feats[feats.year >= 2010]
    print("\nfeatures >= 2010:", len(m))
    # correlación señal: elo_diff debería separar H vs A
    print("elo_diff medio por resultado:",
          m.groupby("result").elo_diff.mean().round(1).to_dict())
    print("cols:", [c for c in feats.columns if c.startswith(("elo", "home_", "away_", "diff_", "h2h"))])


if __name__ == "__main__":
    main()
