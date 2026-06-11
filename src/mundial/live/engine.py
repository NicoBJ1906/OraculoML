"""LiveEngine: PredictionEngine + estado del torneo + corrección online.

Construcción:
1. Replay del histórico completo (idéntico al engine base).
2. Replay de los partidos en vivo EN ORDEN: antes de aplicar cada uno se
   predice con el estado de ese momento (hooks apagados => predicción del
   modelo base puro) y se registra en el OnlineCorrector — así el corrector
   aprende de predicciones honestas, sin leakage.
3. Se cargan tarjetas/lesiones al TournamentState y se ajustan los factores.

En predicción, los hooks del engine base aplican:
- elo_for        -> Elo base + momentum - sanciones/lesiones
- _adjust_lambdas-> gamma de goles del torneo y efecto altitud
- _adjust_probs  -> corrección de frecuencia de empates
"""
from __future__ import annotations

import pandas as pd

from mundial.features.elo import BASE
from mundial.live.online import OnlineCorrector
from mundial.live.state import TournamentState
from mundial.live.store import LiveStore
from mundial.predict.engine import PredictionEngine


class LiveEngine(PredictionEngine):

    def __init__(self, matches: pd.DataFrame, clf, pois_home, pois_away,
                 rho: float, blend: float, store: LiveStore, xgb=None,
                 weights=None, squad_values=None):
        self.state = TournamentState()
        self.corrector = OnlineCorrector()
        # auditoría del torneo: predicción honesta PRE-partido de cada
        # resultado live (la misma que alimenta al corrector) vs lo real
        self.live_audit: list[dict] = []
        self._live_ready = False          # hooks apagados durante el replay
        super().__init__(matches, clf, pois_home, pois_away, rho, blend,
                         xgb=xgb, weights=weights, squad_values=squad_values)

        live = store.results()
        if len(live):
            live = live.sort_values("date", kind="stable")
        for r in live.itertuples(index=False):
            hs, as_ = int(r.home_score), int(r.away_score)
            neutral = _to_bool(r.neutral)
            # predicción base PRE-partido para el online learning
            d = self.match_distribution(r.date, r.home_team, r.away_team,
                                        neutral)
            self.corrector.add_record(
                r.date, d["lambda_home"], d["lambda_away"],
                float(d["probs"]["D"]), hs, as_,
                xg_home=r.xg_home, xg_away=r.xg_away)
            self.live_audit.append({
                "date": pd.Timestamp(r.date),
                "home_team": r.home_team, "away_team": r.away_team,
                "home_score": hs, "away_score": as_,
                "p_home": float(d["probs"]["H"]),
                "p_draw": float(d["probs"]["D"]),
                "p_away": float(d["probs"]["A"])})
            self.state.record_match(
                r.date, r.home_team, r.away_team, hs, as_, neutral,
                self.elo.get(r.home_team, BASE),
                self.elo.get(r.away_team, BASE),
                xg_home=r.xg_home, xg_away=r.xg_away)
            self.apply_result(r.date, r.home_team, r.away_team,
                              hs, as_, neutral, tournament="FIFA World Cup")

        self.state.load_context(store.discipline(), store.injuries())
        self.corrector.fit()
        self._live_ready = True

    # ------------------------------------------------ hooks
    def elo_for(self, team: str, date=None) -> float:
        base = self.elo.get(team, BASE)
        if not self._live_ready or date is None:
            return base
        return base + self.state.adjustment(team, date)

    def _adjust_lambdas(self, date, home, away, lh, la, city=None):
        if not self._live_ready:
            return lh, la
        return self.corrector.adjust_lambdas(lh, la, city)

    def _adjust_probs(self, date, p):
        if not self._live_ready:
            return p
        return self.corrector.adjust_probs(p)

    def live_summary(self) -> dict:
        return self.corrector.summary()


def _to_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1")
    return bool(x)
