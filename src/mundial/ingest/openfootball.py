"""Ingesta de datos del Mundial 2026 (openfootball/worldcup.json, dominio público).

Equipos clasificados, sedes/estadios (para derivar altitud y coordenadas) y
calendario del torneo. Sin API key.
"""
from __future__ import annotations

from mundial.config import RAW, SETTINGS
from mundial.ingest._download import download_files

DEST = RAW / "worldcup2026"


def download():
    cfg = SETTINGS["sources"]["worldcup_2026"]
    return download_files(cfg["base"], cfg["files"], DEST)
