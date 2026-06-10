"""Tests de RBAC (spec §8) y plantillas normalizadas (spec §2.4)."""
import pandas as pd

from mundial.auth import verify_password
from mundial.ingest.rosters import build_rosters


# ------------------------------------------------ RBAC
def test_password_correcta():
    assert verify_password("s3creta", "s3creta")


def test_password_incorrecta():
    assert not verify_password("otra", "s3creta")


def test_fail_closed_sin_secret():
    """Sin secrets configurados NADIE es admin (fail-closed)."""
    assert not verify_password("loquesea", None)
    assert not verify_password("", "s3creta")
    assert not verify_password("", None)


# ------------------------------------------------ rosters
def _gs(rows):
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "team", "scorer", "minute",
                                       "own_goal", "penalty"])


def test_build_rosters_filtra_y_ordena():
    gs = _gs([
        ("2024-01-01", "x", "y", "Mexico", "Gimenez", 10, False, False),
        ("2025-01-01", "x", "y", "Mexico", "Gimenez", 20, False, False),
        ("2025-02-01", "x", "y", "Mexico", "Jimenez", 30, False, False),
        ("2019-01-01", "x", "y", "Mexico", "Viejo", 40, False, False),
        ("2025-03-01", "x", "y", "Noclasificado", "Otro", 50, False, False),
        ("2025-04-01", "x", "y", "Mexico", "Rival", 60, True, False),
    ])
    out = build_rosters(gs, ["Mexico"], since="2022-06-01")
    assert list(out.team.unique()) == ["Mexico"]       # solo clasificados
    assert "Viejo" not in set(out.player)              # fuera por fecha
    assert "Rival" not in set(out.player)              # autogol excluido
    assert list(out.player) == ["Gimenez", "Jimenez"]  # orden por goles
    assert out.iloc[0].goals == 2
