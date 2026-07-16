"""Matemática pura del dashboard de definición (tab "Final", spec §10).

Con solo 2 partidos restantes (Final y tercer puesto) el espacio de
desenlaces es enumerable: aquí NO hay Monte Carlo. Las probabilidades de
cruce llegan ya cocinadas del `PredictionEngine` (`p_home_advances`
incluye prórroga/penales); estas funciones solo componen ese espacio y
la carrera de la Bota de Oro. Puras a propósito: se testean sin
Streamlit ni artefactos (spec §10-F1/F3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Truncamiento de la Poisson de goles restantes por jugador. Con lambdas
# reales (< 2) la masa más allá de 12 goles es ~0; el residuo se suma al
# último bin para que cada pmf siga sumando exactamente 1.
MAX_EXTRA_GOALS = 12


def podium_probs(final_teams: tuple[str, str], p_final_home: float,
                 third_teams: tuple[str, str],
                 p_third_home: float) -> pd.DataFrame:
    """Matriz equipo × puesto {p1..p4}, exacta (F1).

    Los finalistas solo pueden ser 1.º o 2.º; los del tercer puesto solo
    3.º o 4.º — cada fila y cada columna-puesto suman 1.
    """
    (fa, fb), (ta, tb) = final_teams, third_teams
    q_final, q_third = 1.0 - p_final_home, 1.0 - p_third_home
    rows = [
        {"team": fa, "p1": p_final_home, "p2": q_final, "p3": 0.0, "p4": 0.0},
        {"team": fb, "p1": q_final, "p2": p_final_home, "p3": 0.0, "p4": 0.0},
        {"team": ta, "p1": 0.0, "p2": 0.0, "p3": p_third_home, "p4": q_third},
        {"team": tb, "p1": 0.0, "p2": 0.0, "p3": q_third, "p4": p_third_home},
    ]
    return pd.DataFrame(rows)


def podium_scenarios(final_teams: tuple[str, str], p_final_home: float,
                     third_teams: tuple[str, str],
                     p_third_home: float) -> list[dict]:
    """Los 4 desenlaces posibles (ganador de la Final ⊗ ganador del 3.er
    puesto, independientes) con su probabilidad conjunta, ordenados desc.

    Cada escenario: {"p": float, "podium": [1.º, 2.º, 3.º, 4.º]}.
    """
    (fa, fb), (ta, tb) = final_teams, third_teams
    out = []
    for win_f, p_f in ((fa, p_final_home), (fb, 1.0 - p_final_home)):
        lose_f = fb if win_f == fa else fa
        for win_t, p_t in ((ta, p_third_home), (tb, 1.0 - p_third_home)):
            lose_t = tb if win_t == ta else ta
            out.append({"p": p_f * p_t,
                        "podium": [win_f, lose_f, win_t, lose_t]})
    out.sort(key=lambda s: -s["p"])
    return out


def _pois_pmf(lam: float, kmax: int) -> np.ndarray:
    """pmf Poisson truncada en kmax con el residuo sumado al último bin
    (así la pmf sigue sumando 1 y las CDF de la carrera son coherentes)."""
    if lam <= 0.0:
        pmf = np.zeros(kmax + 1)
        pmf[0] = 1.0
        return pmf
    k = np.arange(kmax + 1, dtype=float)
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, kmax + 1)))))
    pmf = np.exp(k * np.log(lam) - lam - log_fact)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def golden_boot_race(contenders: list[dict],
                     kmax: int = MAX_EXTRA_GOALS) -> pd.DataFrame:
    """Carrera de la Bota de Oro (F3): P(terminar máximo goleador).

    contenders: [{"player", "team", "goals", "lam"}] — `goals` ya
    convertidos, `lam` = goles esperados del jugador en SU partido
    restante (0 si está eliminado → masa puntual).

    Independencia entre jugadores (aproximación declarada en la UI: la
    correlación intra-equipo se desprecia). Devuelve por jugador:
    - p_top_solo   = P(estrictamente por encima de todos)
    - p_top_shared = P(nadie lo supera, empate incluido)
    """
    totals: list[np.ndarray] = []
    for c in contenders:
        pmf = _pois_pmf(float(c.get("lam", 0.0)), kmax)
        t = np.zeros(int(c["goals"]) + kmax + 1)
        t[int(c["goals"]):] = pmf
        totals.append(t)
    tmax = max(len(t) for t in totals)
    totals = [np.pad(t, (0, tmax - len(t))) for t in totals]
    cdfs = [np.cumsum(t) for t in totals]

    rows = []
    for i, c in enumerate(contenders):
        p_solo = p_shared = 0.0
        for t in range(tmax):
            p_i = totals[i][t]
            if p_i == 0.0:
                continue
            below, at_most = 1.0, 1.0
            for j in range(len(contenders)):
                if j == i:
                    continue
                below *= cdfs[j][t - 1] if t > 0 else 0.0
                at_most *= cdfs[j][t]
            p_solo += p_i * below
            p_shared += p_i * at_most
        rows.append({"player": c["player"], "team": c["team"],
                     "goals": int(c["goals"]),
                     "lam": float(c.get("lam", 0.0)),
                     "p_top_solo": p_solo, "p_top_shared": p_shared})
    df = pd.DataFrame(rows).sort_values(
        ["p_top_shared", "goals"], ascending=[False, False], kind="stable")
    return df.reset_index(drop=True)
