"""Ingesta del dataset martj42/international_results (GitHub, dominio público).

Cubre 1872 -> 2026 e incluye TODAS las eliminatorias del Mundial (World Cup
qualification) de las seis confederaciones, además de amistosos y torneos
continentales. Archivos: results, goalscorers, shootouts, former_names.
"""
from __future__ import annotations

from mundial.config import RAW, SETTINGS
from mundial.ingest._download import download_files

DEST = RAW / "international"


def download():
    cfg = SETTINGS["sources"]["international_results"]
    return download_files(cfg["base"], cfg["files"], DEST)
