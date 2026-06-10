"""Tests del desempate FIFA de fase de grupos (spec sección 4):
Pts → DG → GF → head-to-head entre empatados → azar."""
import numpy as np
import pytest

from mundial.predict.montecarlo import rank_group


def test_h2h_desempata_a_iguales():
    """A y B terminan idénticos en Pts/DG/GF, pero A le ganó a B: el
    head-to-head debe poner a A primero, sin importar la semilla."""
    teams = ["A", "B", "C", "D"]
    results = {
        ("A", "B"): (1, 0), ("A", "D"): (0, 1), ("A", "C"): (1, 0),
        ("B", "C"): (1, 0), ("B", "D"): (1, 0), ("C", "D"): (1, 0),
    }
    # A: 6 pts, DG +1, GF 2 | B: 6 pts, DG +1, GF 2  -> H2H: A venció a B
    # C: 3 pts, DG -1, GF 1 | D: 3 pts, DG -1, GF 1  -> H2H: C venció a D
    for seed in range(10):
        order, stats = rank_group(teams, results, np.random.default_rng(seed))
        assert order == ["A", "B", "C", "D"]
    assert stats["A"] == [6, 1, 2]
    assert stats["B"] == [6, 1, 2]


def test_orden_primario_por_puntos_dg_gf():
    teams = ["A", "B", "C", "D"]
    results = {
        ("A", "B"): (3, 0), ("A", "C"): (1, 0), ("A", "D"): (2, 0),
        ("B", "C"): (1, 0), ("B", "D"): (1, 0), ("C", "D"): (1, 0),
    }
    order, stats = rank_group(teams, results, np.random.default_rng(0))
    assert order[0] == "A"                      # 9 pts
    assert order[1] == "B"                      # 6 pts
    assert stats["A"][0] == 9


def test_ignora_partidos_de_otros_grupos():
    teams = ["A", "B", "C", "D"]
    results = {
        ("A", "B"): (1, 0), ("X", "Y"): (5, 0),    # X,Y de otro grupo
    }
    order, stats = rank_group(teams, results, np.random.default_rng(0))
    assert "X" not in stats
    assert order[0] == "A"


def test_empate_circular_no_revienta():
    """Tres equipos empatados también en el mini-grupo H2H: se resuelve
    por azar pero siempre devuelve una permutación válida."""
    teams = ["A", "B", "C", "D"]
    results = {                                  # círculo perfecto A>B>C>A
        ("A", "B"): (1, 0), ("B", "C"): (1, 0), ("C", "A"): (1, 0),
        ("A", "D"): (1, 0), ("B", "D"): (1, 0), ("C", "D"): (1, 0),
    }
    order, _ = rank_group(teams, results, np.random.default_rng(0))
    assert sorted(order) == teams
    assert order[3] == "D"                       # D perdió todo: último fijo
