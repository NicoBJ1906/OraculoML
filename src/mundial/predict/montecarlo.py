"""Simulación Monte Carlo del Mundial 2026 completo.

Simula N torneos: fase de grupos (marcadores muestreados de la matriz de cada
partido, o el resultado REAL si ya fue ingresado), tablas con desempates FIFA
(pts → dif. gol → goles a favor), mejores 8 terceros, bracket oficial de
openfootball (R32 → final, con slots de terceros por restricción de grupos) y
eliminatorias con prórroga/penales aproximados por Elo.

P(campeón) y P(llegar a cada ronda) = frecuencia sobre las N simulaciones.
Como usa el estado del PredictionEngine (Elo/forma con resultados en vivo),
las probabilidades se actualizan solas a medida que se ingresan partidos.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROUNDS = ["R32", "R16", "QF", "SF", "F", "CAMPEON"]
ROUND_OF = {"Round of 32": "R32", "Round of 16": "R16",
            "Quarter-final": "QF", "Semi-final": "SF", "Final": "F"}

# Sede del partido de eliminatoria -> país (para localía de anfitriones)
_MX_CITIES = ("Mexico City", "Guadalajara", "Zapopan", "Monterrey", "Guadalupe")
_CA_CITIES = ("Toronto", "Vancouver")
HOST_OF_COUNTRY = {"MX": "Mexico", "CA": "Canada", "US": "United States"}


def _ground_country(ground: str) -> str:
    if any(c in ground for c in _MX_CITIES):
        return "MX"
    if any(c in ground for c in _CA_CITIES):
        return "CA"
    return "US"


def _acc(stats: dict, h: str, a: str, gh: int, ga: int) -> None:
    stats[h][0] += 3 * (gh > ga) + (gh == ga)
    stats[a][0] += 3 * (ga > gh) + (gh == ga)
    stats[h][1] += gh - ga
    stats[a][1] += ga - gh
    stats[h][2] += gh
    stats[a][2] += ga


def rank_group(teams: list[str], results: dict[tuple, tuple],
               rng) -> tuple[list[str], dict]:
    """Ordena un grupo con el desempate FIFA completo:
    Pts → DG → GF → head-to-head entre los empatados (Pts/DG/GF del
    mini-grupo) → azar (proxy de fair play / sorteo).

    Función pura (testeable): `results` es {(home, away): (gh, ga)} y puede
    contener partidos de otros grupos (se ignoran).
    """
    stats = {t: [0, 0, 0] for t in teams}        # pts, dg, gf
    for (h, a), (gh, ga) in results.items():
        if h in stats and a in stats:
            _acc(stats, h, a, gh, ga)

    def key(t: str) -> tuple:
        return (stats[t][0], stats[t][1], stats[t][2])

    primary = sorted(teams, key=key, reverse=True)
    order: list[str] = []
    i = 0
    while i < len(primary):                       # bloques de empate exacto
        j = i + 1
        while j < len(primary) and key(primary[j]) == key(primary[i]):
            j += 1
        block = primary[i:j]
        if len(block) > 1:
            sub = {t: [0, 0, 0] for t in block}   # mini-tabla solo entre ellos
            for (h, a), (gh, ga) in results.items():
                if h in sub and a in sub:
                    _acc(sub, h, a, gh, ga)
            block = sorted(block, key=lambda t: (sub[t][0], sub[t][1],
                                                 sub[t][2], rng.random()),
                           reverse=True)
        order.extend(block)
        i = j
    return order, stats


class TournamentSimulator:
    """Simula el torneo completo desde el estado actual del engine."""

    def __init__(self, engine, fixtures: pd.DataFrame, live: pd.DataFrame,
                 ko_template: list[dict], groups: dict[str, list[str]]):
        """
        fixtures   : 72 partidos de grupos (date, home_team, away_team,
                     neutral, group) — el calendario completo.
        live       : resultados ya ingresados (puede incluir KO manuales).
        ko_template: partidos sin 'group' de worldcup.json (R32 → final).
        groups     : grupo -> lista de 4 selecciones.
        """
        self.engine = engine
        self.fixtures = fixtures.reset_index(drop=True)
        self.groups = groups
        self.ko = sorted(
            [m for m in ko_template if m["round"] != "Match for third place"],
            key=lambda m: m.get("num", 999))

        # resultados reales ya conocidos: grupos (home, away) -> (gh, ga)
        # y eliminatorias (frozenset de equipos) -> ganador
        self.actual: dict[tuple, tuple] = {}
        self.ko_actual: dict[frozenset, str] = {}
        if len(live):
            keys = set(zip(fixtures.home_team, fixtures.away_team))
            for r in live.itertuples(index=False):
                if (r.home_team, r.away_team) in keys:
                    self.actual[(r.home_team, r.away_team)] = (
                        int(r.home_score), int(r.away_score))
                else:                       # partido de eliminatoria ya jugado
                    hs, as_ = int(r.home_score), int(r.away_score)
                    w = getattr(r, "ko_winner", None)
                    if not isinstance(w, str) or not w:
                        w = (r.home_team if hs > as_
                             else r.away_team if as_ > hs else None)
                    if w:
                        self.ko_actual[
                            frozenset((r.home_team, r.away_team))] = w

        # distribución de cada fixture de grupos (cacheada una sola vez)
        self._cum: dict[tuple, np.ndarray] = {}
        self._size = 0
        for r in self.fixtures.itertuples(index=False):
            key = (r.home_team, r.away_team)
            if key in self.actual:
                continue
            d = engine.match_distribution(r.date, r.home_team, r.away_team,
                                          bool(r.neutral),
                                          city=getattr(r, "city", None))
            self._size = d["matrix"].shape[0]
            self._cum[key] = d["matrix"].ravel().cumsum()

        self._precompute_ko_pairs()

    def _precompute_ko_pairs(self, date: str = "2026-07-04") -> None:
        """P(local avanza) para TODOS los pares posibles de eliminatoria,
        en batch (una sola pasada de sklearn — clave para la velocidad)."""
        from mundial.models.baseline import FEATURES
        from mundial.models.poisson import (
            POISSON_FEATURES, outcome_probs, score_matrix,
        )

        teams = [t for ts in self.groups.values() for t in ts]
        hosts = set(HOST_OF_COUNTRY.values())
        rows, keys = [], []
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append(self.engine.features_for(date, h, a, neutral=True))
                keys.append((h, a, True))
                if h in hosts:   # variante con localía (anfitrión en su país)
                    rows.append(self.engine.features_for(date, h, a,
                                                         neutral=False))
                    keys.append((h, a, False))
        X = pd.concat(rows, ignore_index=True)
        p_clf = self.engine.clf.predict_proba(X[FEATURES])      # A, D, H
        lh = self.engine.pois_home.predict(X[POISSON_FEATURES])
        la = self.engine.pois_away.predict(X[POISSON_FEATURES])

        self._p_adv: dict[tuple, float] = {}
        for k, (h, a, _) in enumerate(keys):
            pp = outcome_probs(score_matrix(lh[k], la[k], self.engine.rho))
            p = (self.engine.blend * p_clf[k]
                 + (1 - self.engine.blend) * np.array([pp["A"], pp["D"], pp["H"]]))
            tb = 1.0 / (1.0 + 10 ** ((self.engine.elo_for(a, date)
                                      - self.engine.elo_for(h, date)) / 400.0))
            self._p_adv[keys[k]] = float(p[2] + p[1] * tb)

    # ------------------------------------------------ helpers
    @staticmethod
    def _mkey(i: int, m: dict) -> str:
        return str(m.get("num", f"x{i}"))

    def _sample_score(self, key: tuple, rng) -> tuple[int, int]:
        if key in self.actual:
            return self.actual[key]
        idx = int(np.searchsorted(self._cum[key], rng.random()))
        return divmod(min(idx, self._size * self._size - 1), self._size)

    def _p_first_advances(self, home: str, away: str, ground: str) -> float:
        """P(team1 avanza) con localía si un anfitrión juega en su país."""
        host = HOST_OF_COUNTRY[_ground_country(ground)]
        if home == host:
            return self._p_adv[(home, away, False)]
        if away == host:
            return 1.0 - self._p_adv[(away, home, False)]
        return self._p_adv[(home, away, True)]

    def _standings(self, results: dict[tuple, tuple], rng) -> tuple[dict, list]:
        """1.º/2.º por grupo (desempate FIFA con H2H, ver rank_group) +
        terceros ordenados por Pts → DG → GF (sin H2H entre grupos)."""
        pos: dict[str, str] = {}      # '1A' -> equipo
        thirds: list[tuple] = []
        for g, teams in self.groups.items():
            order, stats = rank_group(teams, results, rng)
            pos[f"1{g}"] = order[0]
            pos[f"2{g}"] = order[1]
            thirds.append((stats[order[2]][0], stats[order[2]][1],
                           stats[order[2]][2], rng.random(), g, order[2]))
        thirds.sort(reverse=True)
        return pos, thirds

    def _assign_thirds(self, slots: list[tuple], thirds: list[tuple],
                       pos: dict) -> None:
        """Asigna los 8 mejores terceros a los slots '3X/Y/Z' del bracket.
        Greedy por ranking respetando los grupos permitidos de cada slot."""
        best8 = thirds[:8]
        used: set[str] = set()
        for code, allowed in slots:
            pick = next((t for t in best8
                         if t[4] in allowed and t[4] not in used), None)
            if pick is None:                      # sin candidato válido:
                pick = next(t for t in best8 if t[4] not in used)
            used.add(pick[4])
            pos[code] = pick[5]

    # ------------------------------------------------ simulación
    def run(self, n_sims: int = 3000, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        teams = [t for ts in self.groups.values() for t in ts]
        reach = {t: defaultdict(int) for t in teams}
        # ocupantes y ganador de cada llave del bracket (para la UI Cuadro)
        slot_stats = {self._mkey(i, m): {"t1": Counter(), "t2": Counter(),
                                         "w": Counter()}
                      for i, m in enumerate(self.ko)}

        third_slots = [(m["team1"], re.findall(r"[A-L]", m["team1"][1:]))
                       for m in self.ko if m["team1"].startswith("3")]
        third_slots += [(m["team2"], re.findall(r"[A-L]", m["team2"][1:]))
                        for m in self.ko if m["team2"].startswith("3")]

        group_keys = list(zip(self.fixtures.home_team, self.fixtures.away_team))

        for _ in range(n_sims):
            results = {k: self._sample_score(k, rng) for k in group_keys}
            pos, thirds = self._standings(results, rng)
            self._assign_thirds(third_slots, thirds, pos)

            winners: dict[str, str] = {}
            for i, m in enumerate(self.ko):
                t1 = pos.get(m["team1"]) or winners.get(m["team1"])
                t2 = pos.get(m["team2"]) or winners.get(m["team2"])
                rnd = ROUND_OF[m["round"]]
                reach[t1][rnd] += 1
                reach[t2][rnd] += 1
                actual_w = self.ko_actual.get(frozenset((t1, t2)))
                if actual_w is not None:
                    w = actual_w
                else:
                    win1 = rng.random() < self._p_first_advances(
                        t1, t2, m.get("ground", ""))
                    w = t1 if win1 else t2
                ss = slot_stats[self._mkey(i, m)]
                ss["t1"][t1] += 1
                ss["t2"][t2] += 1
                ss["w"][w] += 1
                num = m.get("num")
                if num is not None:
                    winners[f"W{num}"] = w
                if rnd == "F":
                    reach[w]["CAMPEON"] += 1

        # top-3 candidatos por slot con su frecuencia (serializable)
        self.slot_stats = {
            k: {side: [(t, c / n_sims) for t, c in cnt.most_common(3)]
                for side, cnt in ss.items()}
            for k, ss in slot_stats.items()}

        rows = []
        g_of = {t: g for g, ts in self.groups.items() for t in ts}
        for t in teams:
            rows.append({"team": t, "group": g_of[t],
                         **{r: reach[t][r] / n_sims for r in ROUNDS}})
        df = pd.DataFrame(rows).sort_values("CAMPEON", ascending=False)
        return df.reset_index(drop=True)
