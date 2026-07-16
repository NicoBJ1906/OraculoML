"""Tests de la matemática pura del dashboard de definición (spec §10).

Cubren los contratos F1 (probabilidades exactas que suman 1) y F3
(carrera de la Bota de Oro por convolución de Poisson truncadas):
casos degenerados con forma cerrada verificable a mano.
"""
from __future__ import annotations

import math

import pytest

from mundial.predict.finalists import (
    golden_boot_race, podium_probs, podium_scenarios,
)

FINAL = ("Spain", "Argentina")
THIRD = ("France", "England")


# ------------------------------------------------ podium_probs (F1)
def test_podium_filas_y_puestos_suman_uno():
    df = podium_probs(FINAL, 0.6, THIRD, 0.55).set_index("team")
    for team in df.index:
        assert df.loc[team, ["p1", "p2", "p3", "p4"]].sum() == pytest.approx(1)
    for col in ("p1", "p2", "p3", "p4"):
        assert df[col].sum() == pytest.approx(1)


def test_podium_finalistas_no_pueden_ser_terceros():
    df = podium_probs(FINAL, 0.6, THIRD, 0.55).set_index("team")
    assert df.loc["Spain", "p3"] == 0 and df.loc["Spain", "p4"] == 0
    assert df.loc["France", "p1"] == 0 and df.loc["France", "p2"] == 0
    assert df.loc["Spain", "p1"] == pytest.approx(0.6)
    assert df.loc["Argentina", "p1"] == pytest.approx(0.4)
    assert df.loc["England", "p3"] == pytest.approx(0.45)


# ------------------------------------------------ podium_scenarios (F1)
def test_escenarios_enumeran_todo_el_espacio():
    sc = podium_scenarios(FINAL, 0.6, THIRD, 0.55)
    assert len(sc) == 4
    assert sum(s["p"] for s in sc) == pytest.approx(1)
    # ordenados desc y el modal = favoritos de ambos partidos
    assert sc[0]["p"] == pytest.approx(0.6 * 0.55)
    assert sc[0]["podium"] == ["Spain", "Argentina", "France", "England"]
    # todo podio es una permutación válida: finalistas arriba, 3er puesto abajo
    for s in sc:
        assert set(s["podium"][:2]) == set(FINAL)
        assert set(s["podium"][2:]) == set(THIRD)


# ------------------------------------------------ golden_boot_race (F3)
def _row(df, player):
    return df.set_index("player").loc[player]


def test_boot_degenerado_sin_partidos_restantes():
    df = golden_boot_race([
        {"player": "A", "team": "X", "goals": 8, "lam": 0.0},
        {"player": "B", "team": "Y", "goals": 6, "lam": 0.0},
        {"player": "C", "team": "Z", "goals": 7, "lam": 0.0},
    ])
    assert _row(df, "A")["p_top_solo"] == pytest.approx(1)
    assert _row(df, "A")["p_top_shared"] == pytest.approx(1)
    assert _row(df, "B")["p_top_shared"] == pytest.approx(0)
    assert _row(df, "C")["p_top_solo"] == pytest.approx(0)


def test_boot_persecucion_forma_cerrada():
    """A lidera 8-7 sin partidos; B juega con lam=1. Forma cerrada:
    P(B comparte o supera) = P(X>=1) = 1-e^-1; P(B en solitario) =
    P(X>=2) = 1-2e^-1; P(A en solitario) = P(X=0) = e^-1."""
    df = golden_boot_race([
        {"player": "A", "team": "X", "goals": 8, "lam": 0.0},
        {"player": "B", "team": "Y", "goals": 7, "lam": 1.0},
    ])
    e = math.exp(-1)
    assert _row(df, "B")["p_top_shared"] == pytest.approx(1 - e, abs=1e-6)
    assert _row(df, "B")["p_top_solo"] == pytest.approx(1 - 2 * e, abs=1e-6)
    assert _row(df, "A")["p_top_solo"] == pytest.approx(e, abs=1e-6)
    assert _row(df, "A")["p_top_shared"] == pytest.approx(2 * e, abs=1e-6)
    # exactamente uno queda en solitario o hay empate en la cima
    p_tie = 1 - _row(df, "A")["p_top_solo"] - _row(df, "B")["p_top_solo"]
    assert p_tie == pytest.approx(e, abs=1e-6)


def test_boot_simetria_y_orden():
    df = golden_boot_race([
        {"player": "A", "team": "X", "goals": 8, "lam": 0.7},
        {"player": "B", "team": "Y", "goals": 8, "lam": 0.7},
        {"player": "C", "team": "Y", "goals": 5, "lam": 0.7},
    ])
    a, b, c = _row(df, "A"), _row(df, "B"), _row(df, "C")
    assert a["p_top_shared"] == pytest.approx(b["p_top_shared"], abs=1e-9)
    assert a["p_top_solo"] == pytest.approx(b["p_top_solo"], abs=1e-9)
    assert a["p_top_solo"] < a["p_top_shared"]          # compartir es más fácil
    assert c["p_top_shared"] < 0.05                      # 3 goles abajo, casi 0
    # los "en solitario" son excluyentes: su suma no puede pasar de 1
    assert df["p_top_solo"].sum() <= 1 + 1e-9
    # orden estable por p_top_shared desc
    assert list(df.player[:2]) == ["A", "B"]
