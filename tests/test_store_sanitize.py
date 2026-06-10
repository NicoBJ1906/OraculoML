"""Invariante S3 del spec: LiveStore sanitiza texto libre en el boundary
de persistencia (anti CSV-injection, sin caracteres de control)."""
import pandas as pd

from mundial.live.store import LiveStore, sanitize_text


def test_sanitize_quita_prefijos_de_formula():
    # Excel/Sheets ejecutan celdas que empiezan con = + - @
    assert sanitize_text("=HYPERLINK('http://evil')") == "HYPERLINK('http://evil')"
    assert sanitize_text("+SUM(A1)") == "SUM(A1)"
    assert sanitize_text("@cmd") == "cmd"
    assert sanitize_text("-2+3") == "2+3"


def test_sanitize_quita_control_y_acota():
    assert sanitize_text("Mes\x00si\r\n") == "Messi"
    assert len(sanitize_text("x" * 500)) == 120


def test_sanitize_preserva_nombres_normales():
    assert sanitize_text("  Kylian Mbappé ") == "Kylian Mbappé"
    assert sanitize_text("O'Brien-Smith") == "O'Brien-Smith"  # guion interno OK


def test_add_match_sanitiza_jugadores_y_formacion(tmp_path):
    store = LiveStore(tmp_path)
    store.add_match(
        {"date": pd.Timestamp("2026-06-11"), "home_team": "Mexico",
         "away_team": "South Africa", "home_score": 1, "away_score": 0,
         "neutral": False, "stage": "group",
         "formation_home": "=4-3-3\x07", "formation_away": "4-4-2"},
        players=[{"team": "Mexico", "player": "=cmd|'/c calc'!A0",
                  "event": "goal", "minute": 10}],
        cards=[{"team": "Mexico", "player": "@evil\x00", "card": "yellow",
                "minute": 50}],
        injuries=[{"team": "Mexico", "player": "+inject",
                   "severity": "next_match"}])
    res = store.results().iloc[0]
    assert res["formation_home"] == "4-3-3"
    assert store.players().iloc[0]["player"] == "cmd|'/c calc'!A0"
    assert store.discipline().iloc[0]["player"] == "evil"
    assert store.injuries().iloc[0]["player"] == "inject"
