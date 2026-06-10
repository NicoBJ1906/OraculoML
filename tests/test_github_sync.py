"""Tests para el módulo de sincronización GitHub (github_sync.py).

No hacen llamadas HTTP reales — parchean urllib para probar el contrato
de la función sin necesitar red ni credenciales.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from mundial.live.github_sync import sync_live_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(body: dict) -> MagicMock:
    """Simula el objeto de respuesta de urllib.request.urlopen."""
    mock = MagicMock()
    mock.read.return_value = json.dumps(body).encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# Secuencia de respuestas mínima que el sync necesita para completar:
# GET ref → GET commit → POST blob × N → POST tree → POST commit → PATCH ref
_HAPPY_PATH_RESPONSES = [
    {"object": {"sha": "abc123"}},         # GET ref
    {"tree": {"sha": "tree000"}},           # GET commit
    {"sha": "blob001"},                     # POST blob (archivo 1)
    {"sha": "new_tree_sha"},               # POST tree
    {"sha": "new_commit_sha1234567"},      # POST commit
    {},                                     # PATCH ref
]


# ---------------------------------------------------------------------------
# Test: flujo nominal
# ---------------------------------------------------------------------------

def test_sync_happy_path_hace_commit(tmp_path: Path, caplog) -> None:
    """Un archivo existente debe completar el happy path sin warnings."""
    csv = tmp_path / "live_results.csv"
    csv.write_text("match_id,date\n20260611_a_b,2026-06-11\n")

    responses = iter(_HAPPY_PATH_RESPONSES)

    with patch("urllib.request.urlopen",
               side_effect=lambda req, timeout: _fake_response(next(responses))):
        with caplog.at_level(logging.INFO, logger="mundial.live.github_sync"):
            sync_live_files([csv], token="ghp_test", repo="owner/repo")

    assert any("github_sync OK" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: archivos inexistentes → no hace llamadas HTTP
# ---------------------------------------------------------------------------

def test_sync_omite_archivos_inexistentes(tmp_path: Path) -> None:
    """Si ningún archivo existe, no debe hacer ninguna llamada HTTP."""
    with patch("urllib.request.urlopen") as mock_url:
        sync_live_files(
            [tmp_path / "no_existe.csv"],
            token="ghp_test",
            repo="owner/repo",
        )
    mock_url.assert_not_called()


# ---------------------------------------------------------------------------
# Test: error HTTP → warning sin excepción
# ---------------------------------------------------------------------------

def test_sync_http_error_loguea_warning_y_no_lanza(tmp_path: Path, caplog) -> None:
    """Un HTTP 401 debe producir WARNING pero nunca propagar la excepción."""
    import urllib.error

    csv = tmp_path / "live_results.csv"
    csv.write_text("x\n")

    error = urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=401,
        msg="Unauthorized",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=error):
        with caplog.at_level(logging.WARNING, logger="mundial.live.github_sync"):
            sync_live_files([csv], token="ghp_invalid", repo="owner/repo")

    assert any("HTTP 401" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: error de red → warning sin excepción
# ---------------------------------------------------------------------------

def test_sync_error_de_red_loguea_warning_y_no_lanza(tmp_path: Path, caplog) -> None:
    """Un timeout / error de conexión debe producir WARNING, no crash."""
    csv = tmp_path / "live_results.csv"
    csv.write_text("x\n")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with caplog.at_level(logging.WARNING, logger="mundial.live.github_sync"):
            sync_live_files([csv], token="ghp_test", repo="owner/repo")

    assert any("github_sync falló" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: LiveStore.add_match llama _sync cuando hay token
# ---------------------------------------------------------------------------

def test_store_add_match_invoca_sync_con_token(tmp_path: Path) -> None:
    """LiveStore debe delegar a sync_live_files cuando hay github_token."""
    from mundial.live.store import LiveStore
    import pandas as pd

    store = LiveStore(tmp_path, github_token="ghp_test",
                      github_repo="owner/repo", github_branch="main")

    with patch("mundial.live.store.sync_live_files") as mock_sync:
        store.add_match({
            "date": pd.Timestamp("2026-06-11"),
            "home_team": "Mexico", "away_team": "South Africa",
            "home_score": 1, "away_score": 0,
            "neutral": False, "stage": "group",
        })

    mock_sync.assert_called_once()
    call_kwargs = mock_sync.call_args
    assert call_kwargs.kwargs["token"] == "ghp_test"
    assert call_kwargs.kwargs["repo"] == "owner/repo"


# ---------------------------------------------------------------------------
# Test: LiveStore.add_match sin token no llama sync
# ---------------------------------------------------------------------------

def test_store_add_match_sin_token_no_invoca_sync(tmp_path: Path) -> None:
    """Sin github_token el store no debe llamar a sync (comportamiento local)."""
    from mundial.live.store import LiveStore
    import pandas as pd

    store = LiveStore(tmp_path)  # sin token

    with patch("mundial.live.store.sync_live_files") as mock_sync:
        store.add_match({
            "date": pd.Timestamp("2026-06-11"),
            "home_team": "Mexico", "away_team": "South Africa",
            "home_score": 1, "away_score": 0,
            "neutral": False, "stage": "group",
        })

    mock_sync.assert_not_called()
