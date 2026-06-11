"""Sincroniza data/live/*.csv al repositorio GitHub tras cada escritura.

Usa la Git Trees API para crear un único commit atómico con los N archivos,
en lugar de N llamadas al Contents API. Ventajas:
- Un solo commit en el historial por partido ingresado
- Sin race conditions entre archivos del mismo evento
- Streamlit Cloud detecta el push y redespliega (~60 s)

Si GitHub no está disponible, falla silenciosamente: los datos ya quedaron
guardados en el CSV local del servidor Cloud. El admin puede reintentar o
el próximo `add_match` intentará sincronizar de nuevo.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

_LOG = logging.getLogger("mundial.live.github_sync")
_API_BASE = "https://api.github.com"
_TIMEOUT = 20  # segundos; la API suele responder en <2 s


def _request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    """Llamada autenticada a la GitHub REST API; lanza HTTPError en ≥4xx."""
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    # nosec B310 — URL siempre construida desde _API_BASE (https://api.github.com),
    # nunca desde input del usuario; no hay riesgo de file:// ni esquema arbitrario.
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # nosec B310
        return json.loads(resp.read())


def sync_live_files(
    files: list[Path],
    token: str,
    repo: str,
    branch: str = "main",
    repo_prefix: str = "data/live",
) -> bool:
    """Sube `files` al repo en un único commit atómico vía Git Trees API.

    Archivos que no existen localmente se omiten (nunca borra el remoto).
    Cualquier error de red o API se registra como WARNING sin lanzar
    excepción — el flujo de ingesta nunca debe fallar por indisponibilidad
    de GitHub. Devuelve True si el commit se publicó (la UI usa el False
    para avisar que el dato quedó solo local).

    Args:
        files:       Rutas locales a subir (normalmente los 4 CSVs de LiveStore).
        token:       GitHub PAT con permiso ``contents:write`` en el repo.
        repo:        ``"owner/repo"`` (ej. ``"NicoBJ1906/OraculoML"``).
        branch:      Rama destino (normalmente ``"main"``).
        repo_prefix: Ruta dentro del repo donde viven los archivos.
    """
    existing = [f for f in files if f.exists()]
    if not existing:
        _LOG.debug("github_sync: ningún archivo existe aún, omitiendo")
        return True

    base = f"{_API_BASE}/repos/{repo}"

    try:
        # 1. SHA del commit HEAD de la rama
        ref_data = _request("GET", f"{base}/git/refs/heads/{branch}", token)
        head_sha: str = ref_data["object"]["sha"]

        # 2. SHA del tree del commit HEAD (necesario como base)
        commit_data = _request("GET", f"{base}/git/commits/{head_sha}", token)
        base_tree_sha: str = commit_data["tree"]["sha"]

        # 3. Crear un blob por cada archivo (contenido en base64)
        tree_items: list[dict] = []
        for path in existing:
            content_b64 = base64.b64encode(path.read_bytes()).decode()
            blob = _request(
                "POST", f"{base}/git/blobs", token,
                {"content": content_b64, "encoding": "base64"},
            )
            tree_items.append({
                "path": f"{repo_prefix}/{path.name}",
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            })

        # 4. Nuevo tree sobre el base_tree (archivos no incluidos quedan igual)
        new_tree = _request(
            "POST", f"{base}/git/trees", token,
            {"base_tree": base_tree_sha, "tree": tree_items},
        )

        # 5. Commit que apunta al nuevo tree
        file_names = ", ".join(p.name for p in existing)
        new_commit = _request(
            "POST", f"{base}/git/commits", token,
            {
                "message": f"live: sync {file_names}",
                "tree": new_tree["sha"],
                "parents": [head_sha],
            },
        )

        # 6. Fast-forward la rama al nuevo commit
        _request(
            "PATCH", f"{base}/git/refs/heads/{branch}", token,
            {"sha": new_commit["sha"]},
        )

        _LOG.info(
            "github_sync OK → %s (%d archivo%s, commit %s)",
            repo, len(existing), "s" if len(existing) != 1 else "",
            new_commit["sha"][:7],
        )
        return True

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace") if exc.fp else ""
        _LOG.warning(
            "github_sync HTTP %s — datos guardados localmente. Detalle: %s",
            exc.code, body[:200],
        )
        return False
    except Exception as exc:  # noqa: BLE001
        _LOG.warning(
            "github_sync falló — datos guardados localmente. Causa: %s", exc
        )
        return False
