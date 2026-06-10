"""Tests del LiveStore y de la consolidación histórico+live (invariantes
L1/L2/M2 del spec). Usa tmp_path: nunca toca data/ real."""
import pandas as pd

from mundial.live.store import LiveStore, make_match_id

RESULT = {
    "date": pd.Timestamp("2026-06-11"), "home_team": "Mexico",
    "away_team": "South Africa", "home_score": 2, "away_score": 0,
    "neutral": False, "stage": "group", "xg_home": 1.8, "xg_away": 0.4,
}
PLAYERS = [{"team": "Mexico", "player": "Gimenez", "event": "goal",
            "minute": 23}]
CARDS = [{"team": "South Africa", "player": "X", "card": "red", "minute": 70}]
INJURIES = [{"team": "Mexico", "player": "Alvarez",
             "severity": "next_match"}]


def test_add_match_escribe_los_cuatro_archivos(tmp_path):
    store = LiveStore(tmp_path)
    mid = store.add_match(RESULT, players=PLAYERS, cards=CARDS,
                          injuries=INJURIES)
    assert mid == make_match_id("2026-06-11", "Mexico", "South Africa")
    assert len(store.results()) == 1
    assert len(store.players()) == 1
    assert len(store.discipline()) == 1
    assert len(store.injuries()) == 1
    # los eventos quedan ligados al partido por match_id (invariante L1)
    assert set(store.players().match_id) == {mid}
    assert set(store.discipline().match_id) == {mid}


def test_delete_match_borra_en_cascada(tmp_path):
    store = LiveStore(tmp_path)
    mid = store.add_match(RESULT, players=PLAYERS, cards=CARDS,
                          injuries=INJURIES)
    otro = store.add_match({**RESULT, "date": pd.Timestamp("2026-06-12"),
                            "home_team": "Spain", "away_team": "Norway"})
    store.delete_match(mid)
    assert list(store.results().match_id) == [otro]
    assert len(store.players()) == 0
    assert len(store.discipline()) == 0
    assert len(store.injuries()) == 0


def test_token_cambia_al_escribir(tmp_path):
    store = LiveStore(tmp_path)
    t0 = store.token()
    store.add_match(RESULT)
    assert store.token() != t0          # invariante L2 (clave de cachés)


def _hist_parquet(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score",
                                     "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    p = tmp_path / "matches.parquet"
    df.to_parquet(p, index=False)
    return p


def test_consolidated_une_y_ordena(tmp_path):
    p = _hist_parquet(tmp_path, [
        ("2025-01-01", "Spain", "France", 1, 1, "Friendly", True)])
    store = LiveStore(tmp_path)
    store.add_match(RESULT)
    out = store.consolidated_matches(p)
    assert len(out) == 2
    assert list(out.date) == sorted(out.date)            # cronológico
    live_row = out[out.home_team == "Mexico"].iloc[0]
    assert live_row.tournament == "FIFA World Cup"
    assert live_row.home_score == 2


def test_consolidated_no_duplica_partidos(tmp_path):
    """Si el histórico ya trae el partido live (martj42 se actualiza a
    diario), gana la fila live y no hay duplicados (invariante M2)."""
    p = _hist_parquet(tmp_path, [
        ("2025-01-01", "Spain", "France", 1, 1, "Friendly", True),
        ("2026-06-11", "Mexico", "South Africa", 9, 9, "FIFA World Cup",
         False)])
    store = LiveStore(tmp_path)
    store.add_match(RESULT)
    out = store.consolidated_matches(p)
    assert len(out) == 2                                  # sin duplicar
    mex = out[out.home_team == "Mexico"].iloc[0]
    assert mex.home_score == 2                            # gana la live


def test_consolidated_sin_live_devuelve_historico(tmp_path):
    p = _hist_parquet(tmp_path, [
        ("2025-01-01", "Spain", "France", 1, 1, "Friendly", True)])
    store = LiveStore(tmp_path)
    out = store.consolidated_matches(p)
    assert len(out) == 1
