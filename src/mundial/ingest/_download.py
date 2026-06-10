"""Descarga de archivos a la zona raw (bronze) del data lake local."""
from __future__ import annotations

import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def download_files(base: str, files: list[str], dest: Path) -> list[Path]:
    """Descarga `files` desde `base` hacia `dest`. Devuelve las rutas guardadas."""
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fname in files:
        url = f"{base}/{fname}"
        out = dest / fname
        log.info("GET %s", url)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        log.info("  -> %s (%.0f KB)", out.name, len(resp.content) / 1024)
        saved.append(out)
    return saved
