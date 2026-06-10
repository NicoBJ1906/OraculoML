"""Consolida histórico + resultados en vivo en un parquet nuevo.

NO toca raw/ ni los parquet originales: escribe
data/interim/matches_consolidated.parquet, que puede usarse como entrada
de 02_build_features.py para un reentrenamiento entre fases del torneo.

Uso:
    python scripts/05_consolidate_live.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundial.live.store import LiveStore  # noqa: E402


def main() -> None:
    store = LiveStore(ROOT)
    src = ROOT / "data" / "interim" / "matches.parquet"
    out = ROOT / "data" / "interim" / "matches_consolidated.parquet"
    df = store.consolidated_matches(src)
    n_live = len(store.results())
    df.to_parquet(out, index=False)
    print(f"OK: {len(df):,} partidos ({n_live} en vivo) -> {out}")


if __name__ == "__main__":
    main()
