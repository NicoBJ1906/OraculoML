"""Test de la asignación oficial FIFA de los 8 mejores terceros a los slots
del bracket (spec: THIRD_ALLOCATION). Con la combinación real del Mundial 2026
(terceros eliminados A, C, G, H) cada slot debe recibir el tercero que manda
el reglamento, no el que elegiría el greedy por ranking."""
import types

from mundial.predict.montecarlo import THIRD_ALLOCATION, TournamentSimulator


def test_asignacion_oficial_terceros_2026():
    # 12 grupos; clasifican como terceros B,D,E,F,I,J,K,L (eliminan A,C,G,H)
    groups = {g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"] for g in "ABCDEFGHIJKL"}
    fake = types.SimpleNamespace(groups=groups)

    # thirds: tupla (pts, dg, gf, azar, grupo, equipo). Los 8 que clasifican
    # van primero (mejor ranking); A,C,G,H después (no entran al best8).
    def third(group, pts):
        return (pts, 0, 0, 0.0, group, f"{group}3")
    thirds = [third(g, 9) for g in "BDEFIJKL"] + [third(g, 1) for g in "ACGH"]

    slots = [
        ("3A/B/C/D/F", list("ABCDF")), ("3C/D/F/G/H", list("CDFGH")),
        ("3C/E/F/H/I", list("CEFHI")), ("3E/H/I/J/K", list("EHIJK")),
        ("3B/E/F/I/J", list("BEFIJ")), ("3A/E/H/I/J", list("AEHIJ")),
        ("3E/F/G/I/J", list("EFGIJ")), ("3D/E/I/J/L", list("DEIJL")),
    ]
    pos: dict = {}
    TournamentSimulator._assign_thirds(fake, slots, thirds, pos)

    # cada slot recibe el tercero del grupo que dictó FIFA (cruces reales R32)
    assert pos["3A/B/C/D/F"] == "D3"   # 1E vs Paraguay (D)
    assert pos["3C/D/F/G/H"] == "F3"   # 1I vs Sweden (F)
    assert pos["3C/E/F/H/I"] == "E3"   # 1A vs Ecuador (E)
    assert pos["3E/H/I/J/K"] == "K3"   # 1L vs DR Congo (K)
    assert pos["3B/E/F/I/J"] == "B3"   # 1D vs Bosnia (B)
    assert pos["3A/E/H/I/J"] == "I3"   # 1G vs Senegal (I)
    assert pos["3E/F/G/I/J"] == "J3"   # 1B vs Algeria (J)
    assert pos["3D/E/I/J/L"] == "L3"   # 1K vs Ghana (L)
    # los 8 terceros usados son exactamente los clasificados
    assert sorted(pos.values()) == [f"{g}3" for g in "BDEFIJKL"]


def test_combinacion_desconocida_cae_al_greedy():
    """Si la combinación de eliminados no está en la tabla, no revienta:
    usa el greedy y asigna 8 terceros distintos respetando los allowed."""
    groups = {g: [f"{g}1"] for g in "ABCDEFGHIJKL"}
    fake = types.SimpleNamespace(groups=groups)
    # eliminados B,D,F,H (no está en THIRD_ALLOCATION)
    classd = "ACEGIJKL"
    assert frozenset("BDFH") not in THIRD_ALLOCATION
    thirds = [(9, 0, 0, 0.0, g, f"{g}3") for g in classd] + \
             [(1, 0, 0, 0.0, g, f"{g}3") for g in "BDFH"]
    slots = [(f"3{g}", list("ACEGIJKL")) for g in "ACEGIJKL"]
    pos: dict = {}
    TournamentSimulator._assign_thirds(fake, slots, thirds, pos)
    assert len(set(pos.values())) == 8
