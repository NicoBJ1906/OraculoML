"""Helpers de presentación puros (sin Streamlit) — testeables aislados."""
from __future__ import annotations

# Umbral para anunciar empate como pronóstico mostrado: el 1X2 rara vez marca
# el empate como argmax aunque sea competitivo (sesgo conocido); aquí lo
# hacemos visible sin alterar la lógica del modelo (bracket/MC usan su propio
# argmax). Cosmético.
DRAW_MIN = 0.30
CLOSE_GAP = 0.12


def display_pred(ph: float, pd_: float, pa: float,
                 home: str, away: str) -> tuple[str, str]:
    """Pronóstico MOSTRADO: si el empate es competitivo (>=DRAW_MIN y la
    diferencia local-visita <=CLOSE_GAP) lo anuncia como partido cerrado; si
    no, el favorito. Devuelve (label, clase_css)."""
    if pd_ >= DRAW_MIN and abs(ph - pa) <= CLOSE_GAP:
        return ("Partido cerrado · empate probable", "d")
    if ph >= pa:
        return (f"Gana {home}", "h")
    return (f"Gana {away}", "a")
