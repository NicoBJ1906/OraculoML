"""Sprint 7 - Valor de plantilla por selección y año (Transfermarkt).

Fuente: salimt/football-datasets (GitHub, CSVs públicos del scraping de
Transfermarkt). El valor de plantilla es la señal #1 que tiene el mercado
de apuestas y que el Elo (100% retrospectivo) no ve: plantillas jóvenes en
ascenso (Inglaterra) vs campeones envejecidos.

Metodología:
1. team_id de selección -> país: ciudadanía MODAL de sus internacionales
   (team_details solo trae clubes; las selecciones se infieren).
2. Valor de la selección en el año Y = suma del top-26 de valores de
   mercado vigentes (última valoración <= 1/jul/Y, con <= 18 meses de
   antigüedad) entre los jugadores que registran partidos con ella.
3. Nombres -> canónico martj42 vía TM_COUNTRY_ALIASES.

Salida: data/processed/squad_values.parquet (team, year, squad_value_eur,
log_value). Cobertura útil: 2004+ (Transfermarkt fiable desde ~2004).

Uso:
    python scripts/07_build_squad_values.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mundial.config import PROCESSED, PROJECT_ROOT  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
YEARS = range(2004, 2027)
TOP_N = 26              # tamaño de plantilla FIFA 2026
MAX_AGE_MONTHS = 18     # una valoración más vieja no cuenta (jugador inactivo)
MIN_PLAYERS = 8         # menos jugadores valorados = selección sin cobertura

# Ciudadanía Transfermarkt -> nombre canónico martj42 (results.csv)
TM_COUNTRY_ALIASES = {
    "Cote d'Ivoire": "Ivory Coast",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "United States": "United States",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "DR Congo": "DR Congo",
    "The Gambia": "Gambia",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "St. Kitts & Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Hongkong": "Hong Kong",
    "Chinese Taipei (Taiwan)": "Taiwan",
    "Macedonia": "North Macedonia",
    "Türkiye": "Turkey",
    "Czech Republic": "Czech Republic",
    "Ireland": "Republic of Ireland",
    "Emirates": "United Arab Emirates",
    "Swaziland": "Eswatini",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
LOG = logging.getLogger(__name__)


def national_team_country(perf: pd.DataFrame, profiles: pd.DataFrame) -> pd.Series:
    """team_id -> país por ciudadanía modal de sus internacionales."""
    cit = profiles[["player_id", "citizenship"]].copy()
    # ciudadanía múltiple separada por DOBLE espacio ("A  B"). OJO: la coma
    # NO separa — "Korea, South" es un solo país. La primera ciudadanía es
    # la deportivamente relevante.
    cit["citizenship"] = (cit["citizenship"].astype("string")
                          .str.split(r"  +", regex=True).str[0].str.strip())
    j = perf.merge(cit, on="player_id", how="inner").dropna(subset=["citizenship"])
    mode = j.groupby("team_id")["citizenship"].agg(lambda s: s.mode().iloc[0])
    counts = j.groupby("team_id")["player_id"].nunique()
    return mode[counts >= MIN_PLAYERS]


def main() -> None:
    perf = pd.read_csv(RAW / "player_national_performances.csv",
                       usecols=["player_id", "team_id"])
    profiles = pd.read_csv(RAW / "player_profiles.csv",
                           usecols=["player_id", "citizenship"])
    values = pd.read_csv(RAW / "player_market_value.csv",
                         parse_dates=["date_unix"])
    values = values.rename(columns={"date_unix": "date"}).dropna(subset=["value"])

    country_of = national_team_country(perf, profiles)
    LOG.info("selecciones detectadas: %s", len(country_of))

    # jugadores por selección (canónico)
    perf = perf[perf.team_id.isin(country_of.index)].copy()
    perf["team"] = perf.team_id.map(country_of).replace(TM_COUNTRY_ALIASES)
    players_of = perf.groupby("team")["player_id"].apply(set)

    values = values.sort_values("date", kind="stable")
    rows = []
    for year in YEARS:
        cutoff = pd.Timestamp(f"{year}-07-01")
        floor = cutoff - pd.DateOffset(months=MAX_AGE_MONTHS)
        window = values[(values.date <= cutoff) & (values.date >= floor)]
        # última valoración vigente por jugador
        last = window.groupby("player_id")["value"].last()
        for team, pids in players_of.items():
            vals = last.reindex(list(pids)).dropna()
            if len(vals) < MIN_PLAYERS:
                continue
            top = vals.nlargest(TOP_N)
            rows.append({"team": team, "year": year,
                         "squad_value_eur": float(top.sum()),
                         "n_players": int(len(vals))})

    out = pd.DataFrame(rows)
    out["log_value"] = np.log10(out["squad_value_eur"].clip(lower=1.0))
    out_path = PROCESSED / "squad_values.parquet"
    out.to_parquet(out_path, index=False)

    LOG.info("squad_values: %s filas, %s selecciones, años %s-%s -> %s",
             len(out), out.team.nunique(), out.year.min(), out.year.max(),
             out_path)
    chk = out[out.year == 2026].nlargest(10, "squad_value_eur")
    print("\nTop-10 plantillas 2026 (EUR):")
    for r in chk.itertuples(index=False):
        print(f"  {r.team:20} {r.squad_value_eur/1e6:8.0f} M  "
              f"({r.n_players} jugadores valorados)")


if __name__ == "__main__":
    main()
