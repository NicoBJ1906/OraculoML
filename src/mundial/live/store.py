"""Persistencia de datos en vivo del torneo (data/live/).

Cuatro archivos CSV, separados del histórico (raw/interim NUNCA se pisan):

- live_results.csv    resultado + contexto del partido (xG, clima,
                      formaciones, ganador de penales en eliminatorias)
- live_players.csv    eventos individuales: goal | penalty | own_goal | assist
- live_discipline.csv tarjetas: yellow | red
- live_injuries.csv   lesiones: next_match | tournament

`consolidated_matches()` une el histórico silver con los partidos en vivo
usando el mismo esquema, para re-correr el pipeline de features sin tocar
los datos originales (mismo patrón que tendría una capa Silver en S3).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mundial.live.github_sync import sync_live_files

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_FORMULA_PREFIX = "=+-@\t "


def sanitize_text(value, maxlen: int = 120) -> str:
    """Sanitiza texto libre antes de persistir (invariante S3 del spec):
    sin caracteres de control, sin prefijos de fórmula (anti
    CSV-injection al abrir los CSV en Excel/Sheets) y longitud acotada.
    El escape HTML del display es una capa aparte (esc() en la UI)."""
    s = _CONTROL_RE.sub("", str(value)).strip()
    s = s.lstrip(_FORMULA_PREFIX)
    return s[:maxlen].strip()


RESULT_COLS = [
    "match_id", "date", "home_team", "away_team", "home_score", "away_score",
    "neutral", "stage", "ko_winner", "xg_home", "xg_away", "weather",
    "formation_home", "formation_away",
]
PLAYER_COLS = ["match_id", "date", "team", "player", "event", "minute"]
CARD_COLS = ["match_id", "date", "team", "player", "card", "minute"]
INJURY_COLS = ["match_id", "date", "team", "player", "severity"]
ODDS_COLS = ["match_id", "date", "home_team", "away_team",
             "odd_home", "odd_draw", "odd_away"]


def make_match_id(date, home: str, away: str) -> str:
    d = pd.Timestamp(date).strftime("%Y%m%d")
    def slug(s: str) -> str:
        return s.lower().replace(" ", "-")
    return f"{d}_{slug(home)}_{slug(away)}"


class LiveStore:
    """Lectura/escritura atómica por partido sobre los CSV de data/live/.

    Si se proporciona ``github_token``, cada escritura se sincroniza al repo
    GitHub en un único commit atómico (Opción A de persistencia para
    Streamlit Cloud). Sin token el comportamiento es idéntico al local.
    """

    def __init__(
        self,
        root: Path,
        *,
        github_token: str | None = None,
        github_repo: str = "NicoBJ1906/OraculoML",
        github_branch: str = "main",
    ):
        self.dir = Path(root) / "data" / "live"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f_results = self.dir / "live_results.csv"
        self.f_players = self.dir / "live_players.csv"
        self.f_cards = self.dir / "live_discipline.csv"
        self.f_injuries = self.dir / "live_injuries.csv"
        self.f_odds = self.dir / "live_odds.csv"
        self._github_token = github_token
        self._github_repo = github_repo
        self._github_branch = github_branch
        # estado del último sync (None = sin token / sin intentar): la UI
        # lo usa para avisar cuando el dato quedó solo local
        self.last_sync_ok: bool | None = None

    def _sync(self) -> None:
        """Sincroniza los CSVs live a GitHub si hay token configurado."""
        if not self._github_token:
            return
        self.last_sync_ok = sync_live_files(
            [self.f_results, self.f_players, self.f_cards, self.f_injuries,
             self.f_odds],
            token=self._github_token,
            repo=self._github_repo,
            branch=self._github_branch,
        )

    # ------------------------------------------------ lectura
    def _read(self, path: Path, cols: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(path, parse_dates=["date"])
        for c in cols:           # compat con esquemas viejos
            if c not in df.columns:
                df[c] = pd.NA
        if "match_id" in df.columns and df["match_id"].isna().any():
            df["match_id"] = df.apply(
                lambda r: r.match_id if pd.notna(r.match_id) else
                make_match_id(r.date, r.home_team, r.away_team), axis=1)
        return df[cols]

    def results(self) -> pd.DataFrame:
        return self._read(self.f_results, RESULT_COLS)

    def players(self) -> pd.DataFrame:
        return self._read(self.f_players, PLAYER_COLS)

    def discipline(self) -> pd.DataFrame:
        return self._read(self.f_cards, CARD_COLS)

    def injuries(self) -> pd.DataFrame:
        return self._read(self.f_injuries, INJURY_COLS)

    def odds(self) -> pd.DataFrame:
        return self._read(self.f_odds, ODDS_COLS)

    def add_odds(self, row: dict) -> None:
        """Guarda/actualiza las cuotas 1X2 de un partido (una fila por
        match_id; reingresar reemplaza)."""
        mid = make_match_id(row["date"], row["home_team"], row["away_team"])
        cur = self.odds()
        cur = cur[cur.match_id != mid]
        new = pd.concat([cur, pd.DataFrame([{**row, "match_id": mid}])[
            ODDS_COLS]], ignore_index=True)
        new.to_csv(self.f_odds, index=False)
        self._sync()

    # ------------------------------------------------ escritura
    def _append(self, path: Path, cols: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        cur = self._read(path, cols)
        new = pd.concat([cur, pd.DataFrame(rows)[cols]], ignore_index=True)
        new.to_csv(path, index=False)

    def add_match(self, result: dict, players: list[dict] | None = None,
                  cards: list[dict] | None = None,
                  injuries: list[dict] | None = None) -> str:
        """Guarda un partido completo. Devuelve el match_id.
        Todo texto libre (jugadores, formaciones) se sanitiza aquí —
        boundary único de escritura (invariante S3)."""
        mid = make_match_id(result["date"], result["home_team"],
                            result["away_team"])
        result = {**{c: pd.NA for c in RESULT_COLS}, **result,
                  "match_id": mid}
        for k in ("formation_home", "formation_away", "weather"):
            if pd.notna(result.get(k)):
                result[k] = sanitize_text(result[k], 40)

        def _clean(rows: list[dict] | None) -> list[dict]:
            return [{**r, "player": sanitize_text(r["player"])}
                    if "player" in r else r for r in (rows or [])]

        self._append(self.f_results, RESULT_COLS, [result])
        stamp = {"match_id": mid, "date": result["date"]}
        self._append(self.f_players, PLAYER_COLS,
                     [{**stamp, **p} for p in _clean(players)])
        self._append(self.f_cards, CARD_COLS,
                     [{**stamp, **c} for c in _clean(cards)])
        self._append(self.f_injuries, INJURY_COLS,
                     [{**stamp, **i} for i in _clean(injuries)])
        self._sync()
        return mid

    def delete_match(self, match_id: str) -> None:
        """Borra un partido y todos sus eventos asociados."""
        for path, cols in ((self.f_results, RESULT_COLS),
                           (self.f_players, PLAYER_COLS),
                           (self.f_cards, CARD_COLS),
                           (self.f_injuries, INJURY_COLS),
                           (self.f_odds, ODDS_COLS)):
            if path.exists():
                df = self._read(path, cols)
                df[df.match_id != match_id].to_csv(path, index=False)
        self._sync()

    # ------------------------------------------------ utilidades
    def token(self) -> str:
        """Cambia cuando cambia cualquier archivo live (clave de caché)."""
        return "|".join(str(p.stat().st_mtime_ns) if p.exists() else "0"
                        for p in (self.f_results, self.f_players,
                                  self.f_cards, self.f_injuries, self.f_odds))

    def consolidated_matches(self, matches_parquet: Path) -> pd.DataFrame:
        """Histórico silver + resultados en vivo, mismo esquema, sin duplicar
        (si un partido live ya apareciera en el histórico se queda el live)."""
        hist = pd.read_parquet(matches_parquet)
        live = self.results()
        if live.empty:
            return hist
        add = live[["date", "home_team", "away_team", "home_score",
                    "away_score", "neutral"]].copy()
        add["tournament"] = "FIFA World Cup"
        key = ["date", "home_team", "away_team"]
        merged = pd.concat([hist[~hist.set_index(key).index
                                 .isin(add.set_index(key).index)], add],
                           ignore_index=True)
        return merged.sort_values("date", kind="stable").reset_index(drop=True)
