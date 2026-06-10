"""Configuración central: rutas del proyecto y carga de settings.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


SETTINGS = load_settings()


def _zone(name: str) -> Path:
    """Devuelve la ruta de una zona del data lake, creándola si no existe."""
    path = PROJECT_ROOT / SETTINGS["paths"][name]
    path.mkdir(parents=True, exist_ok=True)
    return path


RAW = _zone("raw")
INTERIM = _zone("interim")
PROCESSED = _zone("processed")
