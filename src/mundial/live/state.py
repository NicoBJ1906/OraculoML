"""Feature State Updating: ajustes Elo en vivo por encima del estado base.

El Elo base del engine se actualiza con K=30 (consistente con el
entrenamiento — NO tocar). Esta capa calcula un AJUSTE aditivo en puntos
Elo que solo se aplica al predecir:

- momentum : actualización extra equivalente a subir K a K_LIVE durante el
  torneo, usando margen efectivo goles/xG (si se ingresó xG). Captura que
  un resultado del Mundial pesa más que un amistoso de hace meses.
- sanciones: roja => fuera el siguiente partido; 2 amarillas acumuladas =>
  un partido (el acumulado se limpia tras cuartos, regla FIFA 2026).
- lesiones : "next_match" (un partido) o "tournament" (resto del torneo).

Las penalizaciones son heurísticas conservadoras (puntos Elo por jugador,
con tope) porque NO hay datos históricos para entrenarlas; el efecto fluye
a las features elo_* del clasificador y del Poisson.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from mundial.features.elo import HFA, K, _g_multiplier

K_LIVE = 45.0          # K efectivo durante el torneo (extra = K_LIVE - K)
XG_BLEND = 0.35        # peso del xG en el margen efectivo del momentum
PEN_SUSPENSION = 9.0   # puntos Elo por jugador suspendido
PEN_INJ_MATCH = 7.0    # lesión que pierde un partido
PEN_INJ_TOURN = 14.0   # lesión que pierde el resto del torneo
MAX_PENALTY = 45.0     # tope total de penalización por equipo
MOMENTUM_CAP = 60.0
YELLOW_RESET = pd.Timestamp("2026-07-11")  # amarillas se limpian tras 4tos


class TournamentState:
    """Acumula el estado en vivo y expone adjustment(team, date)."""

    def __init__(self):
        self.momentum: dict[str, float] = defaultdict(float)
        self.match_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
        # (team, fecha_evento, puntos, etiqueta): aplica al siguiente partido
        self._next_match_pens: list[tuple[str, pd.Timestamp, float, str]] = []
        # (team, fecha_evento, puntos, etiqueta): aplica hasta el final
        self._tournament_pens: list[tuple[str, pd.Timestamp, float, str]] = []

    # ------------------------------------------------ momentum
    def record_match(self, date, home: str, away: str, hs: int, as_: int,
                     neutral: bool, elo_home_pre: float, elo_away_pre: float,
                     xg_home: float | None = None,
                     xg_away: float | None = None) -> None:
        """Registra un partido jugado del torneo (en orden cronológico)."""
        date = pd.Timestamp(date)
        eff_h = float(hs) if _is_na(xg_home) else \
            (1 - XG_BLEND) * hs + XG_BLEND * float(xg_home)
        eff_a = float(as_) if _is_na(xg_away) else \
            (1 - XG_BLEND) * as_ + XG_BLEND * float(xg_away)
        adv = 0.0 if neutral else HFA
        exp_h = 1.0 / (1.0 + 10 ** ((elo_away_pre - elo_home_pre - adv) / 400.0))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        extra = (K_LIVE - K) * _g_multiplier(round(eff_h - eff_a)) * (s_h - exp_h)
        self.momentum[home] = _clip(self.momentum[home] + extra, MOMENTUM_CAP)
        self.momentum[away] = _clip(self.momentum[away] - extra, MOMENTUM_CAP)
        self.match_dates[home].append(date)
        self.match_dates[away].append(date)

    # ------------------------------------------------ sanciones y lesiones
    def load_context(self, discipline: pd.DataFrame,
                     injuries: pd.DataFrame) -> None:
        """Procesa tarjetas y lesiones (DataFrames del LiveStore)."""
        if len(discipline):
            for (team, player), grp in discipline.sort_values("date").groupby(
                    ["team", "player"], sort=False):
                yellows = 0
                prev: pd.Timestamp | None = None
                for r in grp.itertuples(index=False):
                    d = pd.Timestamp(r.date)
                    if prev is not None and prev <= YELLOW_RESET < d:
                        yellows = 0          # se limpia el acumulado tras 4tos
                    if r.card == "red":
                        self._next_match_pens.append(
                            (team, d, PEN_SUSPENSION, f"Roja de {player}"))
                        yellows = 0
                    else:
                        yellows += 1
                        if yellows >= 2:
                            self._next_match_pens.append(
                                (team, d, PEN_SUSPENSION,
                                 f"2 amarillas de {player}"))
                            yellows = 0
                    prev = d
        if len(injuries):
            for r in injuries.itertuples(index=False):
                d = pd.Timestamp(r.date)
                if r.severity == "tournament":
                    self._tournament_pens.append(
                        (r.team, d, PEN_INJ_TOURN,
                         f"Lesión de {r.player} (torneo)"))
                else:
                    self._next_match_pens.append(
                        (r.team, d, PEN_INJ_MATCH,
                         f"Lesión de {r.player} (un partido)"))

    # ------------------------------------------------ consulta
    def _played_between(self, team: str, d0: pd.Timestamp,
                        d1: pd.Timestamp) -> bool:
        """¿El equipo jugó otro partido estrictamente entre d0 y d1?"""
        return any(d0 < d < d1 for d in self.match_dates.get(team, ()))

    def explain(self, team: str, date) -> dict:
        """Desglose XAI del ajuste: momentum + cada sanción/lesión activa
        con su etiqueta y puntos. Contrato de la UI (invariante S2)."""
        date = pd.Timestamp(date)
        items = [(label, -p) for t, d0, p, label in self._tournament_pens
                 if t == team and date > d0]
        items += [(label, -p) for t, d0, p, label in self._next_match_pens
                  if t == team and date > d0
                  and not self._played_between(team, d0, date)]
        penalty = min(sum(-pts for _, pts in items), MAX_PENALTY)
        momentum = self.momentum.get(team, 0.0)
        return {"momentum": momentum, "items": items, "penalty": penalty,
                "total": momentum - penalty}

    def penalty(self, team: str, date) -> float:
        return self.explain(team, date)["penalty"]

    def adjustment(self, team: str, date) -> float:
        """Ajuste total en puntos Elo para `team` en un partido en `date`."""
        return self.explain(team, date)["total"]


def _clip(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


def _is_na(x) -> bool:
    return x is None or pd.isna(x)
