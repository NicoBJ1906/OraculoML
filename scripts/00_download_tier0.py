"""Sprint 1 - Tier-0: descarga de datos base del data lake local.

Fuentes 100% gratuitas y sin API key:
  - martj42/international_results : resultados 1872-2026 (incl. eliminatorias)
  - openfootball/worldcup.json   : equipos, estadios y calendario del Mundial 2026

Uso:
    python scripts/00_download_tier0.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete (añade src/ al path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mundial.ingest import international_results, openfootball  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    saved: list[Path] = []
    saved += international_results.download()
    saved += openfootball.download()

    print(f"\nOK - {len(saved)} archivos descargados:")
    for path in saved:
        size_kb = path.stat().st_size / 1024
        print(f"  - {path}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
