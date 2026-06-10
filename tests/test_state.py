"""Tests del castigo Elo por tarjetas/lesiones (spec: invariante S1/S2).

Cubren: roja => siguiente partido, doble amarilla acumulada, limpieza de
amarillas tras cuartos (YELLOW_RESET), lesiones por gravedad, tope de
penalización, momentum con y sin xG, y el contrato de explain().
"""
import pandas as pd
import pytest

from mundial.live.state import (
    MAX_PENALTY, PEN_INJ_MATCH, PEN_INJ_TOURN, PEN_SUSPENSION,
    TournamentState, YELLOW_RESET,
)

CARD_COLS = ["match_id", "date", "team", "player", "card", "minute"]
INJ_COLS = ["match_id", "date", "team", "player", "severity"]


def cards(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CARD_COLS)


def injuries(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=INJ_COLS)


EMPTY_CARDS = cards([])
EMPTY_INJ = injuries([])


# ------------------------------------------------ sanciones por tarjetas
def test_roja_castiga_solo_el_siguiente_partido():
    st = TournamentState()
    st.load_context(cards([("m1", "2026-06-11", "CZE", "Soucek", "red", 70)]),
                    EMPTY_INJ)
    # activa para cualquier fecha posterior al partido de la roja
    assert st.penalty("CZE", "2026-06-17") == PEN_SUSPENSION
    # no aplica el mismo día del partido donde se la sacaron
    assert st.penalty("CZE", "2026-06-11") == 0
    # tras jugar su siguiente partido, la sanción ya fue cumplida
    st.match_dates["CZE"].append(pd.Timestamp("2026-06-17"))
    assert st.penalty("CZE", "2026-06-22") == 0


def test_una_amarilla_no_castiga_dos_si():
    st = TournamentState()
    st.load_context(cards([("m1", "2026-06-11", "MEX", "Alvarez", "yellow", 30)]),
                    EMPTY_INJ)
    assert st.penalty("MEX", "2026-06-17") == 0

    st2 = TournamentState()
    st2.load_context(
        cards([("m1", "2026-06-11", "MEX", "Alvarez", "yellow", 30),
               ("m2", "2026-06-17", "MEX", "Alvarez", "yellow", 80)]),
        EMPTY_INJ)
    assert st2.penalty("MEX", "2026-06-22") == PEN_SUSPENSION
    # la doble amarilla NO está activa antes de la segunda tarjeta
    assert st2.penalty("MEX", "2026-06-15") == 0


def test_amarillas_se_limpian_tras_cuartos():
    """Regla FIFA: el acumulado de amarillas se borra después de cuartos."""
    assert YELLOW_RESET == pd.Timestamp("2026-07-11")
    st = TournamentState()
    st.load_context(
        cards([("m1", "2026-06-20", "ARG", "Paredes", "yellow", 30),
               ("m2", "2026-07-14", "ARG", "Paredes", "yellow", 60)]),
        EMPTY_INJ)
    # una amarilla antes de cuartos + una después NO suman doble amarilla
    assert st.penalty("ARG", "2026-07-16") == 0

    st2 = TournamentState()
    st2.load_context(
        cards([("m1", "2026-07-14", "ARG", "Paredes", "yellow", 30),
               ("m2", "2026-07-15", "ARG", "Paredes", "yellow", 60)]),
        EMPTY_INJ)
    # dos amarillas después del reset sí suman
    assert st2.penalty("ARG", "2026-07-17") == PEN_SUSPENSION


# ------------------------------------------------ lesiones
def test_lesion_un_partido_se_limpia_al_jugar():
    st = TournamentState()
    st.load_context(EMPTY_CARDS, injuries(
        [("m1", "2026-06-11", "KOR", "Kim", "next_match")]))
    assert st.penalty("KOR", "2026-06-17") == PEN_INJ_MATCH
    st.match_dates["KOR"].append(pd.Timestamp("2026-06-17"))
    assert st.penalty("KOR", "2026-06-22") == 0


def test_lesion_torneo_persiste():
    st = TournamentState()
    st.load_context(EMPTY_CARDS, injuries(
        [("m1", "2026-06-11", "FRA", "Mbappe", "tournament")]))
    st.match_dates["FRA"] += [pd.Timestamp("2026-06-17"),
                              pd.Timestamp("2026-06-22")]
    # sigue activa aunque el equipo juegue más partidos
    assert st.penalty("FRA", "2026-07-10") == PEN_INJ_TOURN


def test_tope_de_penalizacion():
    st = TournamentState()
    st.load_context(
        cards([("m1", "2026-06-11", "QAT", f"Jugador{i}", "red", 50)
               for i in range(6)]),       # 6 rojas x 9 = 54 > tope
        EMPTY_INJ)
    assert st.penalty("QAT", "2026-06-15") == MAX_PENALTY


# ------------------------------------------------ momentum
def test_momentum_ganador_sube_perdedor_baja():
    st = TournamentState()
    st.record_match("2026-06-11", "A", "B", 2, 0, True, 1500.0, 1500.0)
    assert st.momentum["A"] > 0
    assert st.momentum["B"] == pytest.approx(-st.momentum["A"])


def test_momentum_xg_amplifica_margen():
    """Ganar 1-0 con xG 3.0-0.3 da más momentum que ganar 1-0 a secas."""
    sin_xg = TournamentState()
    sin_xg.record_match("2026-06-11", "A", "B", 1, 0, True, 1500.0, 1500.0)
    con_xg = TournamentState()
    con_xg.record_match("2026-06-11", "A", "B", 1, 0, True, 1500.0, 1500.0,
                        xg_home=3.0, xg_away=0.3)
    assert con_xg.momentum["A"] > sin_xg.momentum["A"]


# ------------------------------------------------ XAI (invariante S2)
def test_explain_expone_etiquetas_y_total():
    st = TournamentState()
    st.load_context(
        cards([("m1", "2026-06-11", "CZE", "Soucek", "red", 70)]),
        injuries([("m1", "2026-06-11", "CZE", "Coufal", "next_match")]))
    e = st.explain("CZE", "2026-06-17")
    labels = [lbl for lbl, _ in e["items"]]
    assert any("Roja de Soucek" in lbl for lbl in labels)
    assert any("Lesión de Coufal" in lbl for lbl in labels)
    assert e["penalty"] == PEN_SUSPENSION + PEN_INJ_MATCH
    assert e["total"] == pytest.approx(e["momentum"] - e["penalty"])
