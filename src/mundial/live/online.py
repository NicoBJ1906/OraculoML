"""Online Learning ligero: corrige el modelo base con el torneo en curso.

NO reentrena nada (con n<=104 partidos un fit destruiría la calibración,
que es la fortaleza medida del modelo). En su lugar aplica factores de
corrección con shrinkage bayesiano — con 0 partidos los factores son 1.0
(modelo base intacto) y se mueven gradualmente según acumula evidencia:

- gamma      : ritmo de goles del torneo vs lo predicho (usa xG si se
               ingresó: el xG es menos ruidoso que el marcador).
- draw_mult  : frecuencia de empates observada vs predicha.
- alt_mult   : efecto altitud en sedes >= 1400 m (CDMX 2240 m,
               Guadalajara 1566 m), prior suave 1.05 refinado online.
- ko_temp    : temperatura de afilado SOLO para eliminatorias (O3). Los
               favoritos KO del torneo rinden por encima de su probabilidad
               declarada; se ajusta por log-loss sobre los KO ya jugados
               con shrinkage fuerte hacia 1.0 y nunca toca los grupos.

Todos los factores van capados: la corrección nunca puede volverse más
grande que la señal que la sustenta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Altitud (m) de las sedes 2026, por el nombre de ciudad de results.csv.
ALTITUDE_M = {
    "Mexico City": 2240, "Zapopan": 1566, "Guadalajara": 1566,
    "Guadalupe": 540, "Monterrey": 540, "Kansas City": 270,
    "Atlanta": 320, "Arlington": 150, "Dallas": 150, "Foxborough": 89,
    "Toronto": 76, "Inglewood": 30, "Los Angeles": 30, "Houston": 15,
    "Philadelphia": 12, "Seattle": 5, "Santa Clara": 3,
    "East Rutherford": 3, "Miami Gardens": 3, "Vancouver": 0,
}
ALT_THRESHOLD = 1400

N0_GOALS = 20.0    # prior en goles esperados (~8 partidos de evidencia previa)
K0_DRAW = 12.0     # prior empates (bajado de 25: el Mundial 2026 arrancó con
                   # ~43% de empates, el doble de lo normal — el corrector
                   # debe responder más rápido a esa señal fuerte)
DRAW_CAP = 1.55    # techo de draw_mult (subido de 1.25 por el mismo motivo)
ALT_PRIOR = 1.05   # prior: ~5% más goles en altura
ALT_N0 = 8.0
XG_BLEND = 0.35    # peso del xG en los "goles observados"
KO_T_N0 = 15.0     # prior de la temperatura KO (partidos de evidencia)
KO_T_CLIP = (0.80, 1.10)
KO_T_GRID = np.arange(0.60, 1.21, 0.05)


def altitude_of(city: str | None) -> int:
    return ALTITUDE_M.get(city, 0) if city else 0


class OnlineCorrector:
    """Acumula (predicción base, resultado) y ajusta predicciones futuras."""

    def __init__(self):
        self._rec: list[dict] = []
        self.gamma = 1.0
        self.draw_mult = 1.0
        self.alt_mult = ALT_PRIOR
        self.ko_temp = 1.0
        self.ko_start: pd.Timestamp | None = None
        self.n = 0
        self.n_ko = 0

    def add_record(self, date, lambda_home: float, lambda_away: float,
                   p_draw: float, hs: int, as_: int,
                   xg_home=None, xg_away=None, city: str | None = None,
                   probs: tuple | None = None, stage: str = "group") -> None:
        """Registra un partido jugado con la predicción hecha ANTES de él.

        probs = (pA, pD, pH) pre-partido del ensemble; stage identifica los
        KO ('R32'/'R16'/'QF'/'SF'/'F') para ajustar ko_temp (O3).
        """
        obs_h = float(hs) if _na(xg_home) else \
            (1 - XG_BLEND) * hs + XG_BLEND * float(xg_home)
        obs_a = float(as_) if _na(xg_away) else \
            (1 - XG_BLEND) * as_ + XG_BLEND * float(xg_away)
        self._rec.append({
            "date": pd.Timestamp(date), "pred": lambda_home + lambda_away,
            "obs": obs_h + obs_a, "p_draw": p_draw,
            "draw": int(hs == as_), "alt": altitude_of(city),
            "ko": str(stage).strip().lower() != "group",
            "pvec": probs,
            "out": 2 if hs > as_ else (0 if as_ > hs else 1),  # orden A,D,H
        })

    def fit(self) -> None:
        """Recalcula los factores con todo lo registrado."""
        self.n = len(self._rec)
        if not self.n:
            return
        df = pd.DataFrame(self._rec)

        # ritmo de goles: razón observado/predicho con prior N0_GOALS
        pred, obs = df["pred"].sum(), df["obs"].sum()
        self.gamma = float(np.clip((obs + N0_GOALS) / (pred + N0_GOALS),
                                   0.85, 1.18))

        # empates: razón frecuencia/probabilidad media con prior K0_DRAW
        mean_pd = df["p_draw"].mean()
        if mean_pd > 0:
            self.draw_mult = float(np.clip(
                (df["draw"].sum() + K0_DRAW * mean_pd)
                / ((self.n + K0_DRAW) * mean_pd), 0.80, DRAW_CAP))

        # altitud: solo con los partidos jugados en altura
        hi = df[df["alt"] >= ALT_THRESHOLD]
        if len(hi):
            ratio = hi["obs"].sum() / max(hi["pred"].sum(), 1e-9)
            w = len(hi) / (len(hi) + ALT_N0)
            self.alt_mult = float(np.clip(
                w * ratio + (1 - w) * ALT_PRIOR, 0.90, 1.20))

        # temperatura KO (O3): grid de log-loss sobre los KO jugados,
        # con shrinkage hacia 1.0 — nunca se ajusta con grupos.
        ko = df[df["ko"] & df["pvec"].notna()]
        self.n_ko = len(ko)
        if self.n_ko:
            self.ko_start = ko["date"].min()
            P = np.array([list(v) for v in ko["pvec"]], dtype=float)
            y = ko["out"].to_numpy()
            idx = np.arange(len(y))
            lls = []
            for t in KO_T_GRID:
                q = P ** (1.0 / t)
                q /= q.sum(axis=1, keepdims=True)
                lls.append(-np.log(np.clip(q[idx, y], 1e-9, None)).mean())
            t_fit = float(KO_T_GRID[int(np.argmin(lls))])
            w = self.n_ko / (self.n_ko + KO_T_N0)
            self.ko_temp = float(np.clip(
                w * t_fit + (1 - w) * 1.0, *KO_T_CLIP))

    # ------------------------------------------------ aplicación
    def adjust_lambdas(self, lh: float, la: float,
                       city: str | None = None) -> tuple[float, float]:
        m = self.gamma
        if altitude_of(city) >= ALT_THRESHOLD:
            m *= self.alt_mult
        return lh * m, la * m

    def is_ko(self, date) -> bool:
        """True si la fecha cae en eliminatorias ya iniciadas del torneo."""
        if self.ko_start is None:
            return False
        return bool(pd.Timestamp(date) >= self.ko_start)

    def adjust_probs(self, p: np.ndarray, ko: bool = False) -> np.ndarray:
        """p en orden ['A','D','H']. Reescala P(empate), renormaliza y en
        eliminatorias afila con la temperatura KO (O3)."""
        q = p
        if self.draw_mult != 1.0:
            q = p.copy()
            q[1] = min(q[1] * self.draw_mult, 0.90)
            rest = q[0] + q[2]
            if rest > 0:
                scale = (1.0 - q[1]) / rest
                q[0] *= scale
                q[2] *= scale
            q = q / q.sum()
        if ko and self.ko_temp != 1.0:
            q = q ** (1.0 / self.ko_temp)
            q = q / q.sum()
        return q

    def summary(self) -> dict:
        return {"n": self.n, "gamma": self.gamma,
                "draw_mult": self.draw_mult, "alt_mult": self.alt_mult,
                "ko_temp": self.ko_temp, "n_ko": self.n_ko}


def _na(x) -> bool:
    return x is None or pd.isna(x)
