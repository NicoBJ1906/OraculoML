"""Normalización de nombres de selecciones y jugadores para el data lake.

Canónico = nombre en inglés del dataset martj42 (results.csv). Se aplican:
  1. former_names.csv : nombres históricos -> nombre actual (por rango de fechas).
  2. TEAM_ALIASES     : armoniza otras fuentes (openfootball) hacia el canónico.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

# Variantes de otras fuentes -> nombre canónico (el de martj42).
# Detectado por diagnóstico: solo 2 de los 48 del Mundial 2026 difieren.
TEAM_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    # extensible: añadir aquí cualquier mismatch futuro
}

# Nombres de país de Transfermarkt (citizenship) -> canónico martj42.
# Usado por scripts 06 (rosters) y 07 (valores de plantilla).
TM_COUNTRY_ALIASES: dict[str, str] = {
    "Cote d'Ivoire": "Ivory Coast",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "The Gambia": "Gambia",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "St. Kitts & Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "Hongkong": "Hong Kong",
    "Chinese Taipei (Taiwan)": "Taiwan",
    "Macedonia": "North Macedonia",
    "Türkiye": "Turkey",
    "Ireland": "Republic of Ireland",
    "Emirates": "United Arab Emirates",
    "Swaziland": "Eswatini",
}


def clean_text(s: pd.Series) -> pd.Series:
    """Trim, colapsa espacios internos y normaliza unicode (NFC). Mantiene acentos."""
    s = s.astype("string")
    s = s.str.normalize("NFC")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def normalize_team(s: pd.Series) -> pd.Series:
    """Limpia y aplica alias de selección hacia el nombre canónico en inglés."""
    return clean_text(s).replace(TEAM_ALIASES)


def apply_former_names(
    df: pd.DataFrame, former: pd.DataFrame, col: str, date_col: str = "date"
) -> pd.Series:
    """Reemplaza nombres históricos por el actual según el rango de fechas."""
    out = df[col].copy()
    for row in former.itertuples(index=False):
        mask = (
            (out == row.former)
            & (df[date_col] >= row.start_date)
            & (df[date_col] <= row.end_date)
        )
        out = out.mask(mask, row.current)
    return out


def strip_accents(s: pd.Series) -> pd.Series:
    """Versión ASCII (sin acentos) para joins de jugadores. No reemplaza el original."""

    def _f(x):
        if pd.isna(x):
            return x
        return "".join(
            c
            for c in unicodedata.normalize("NFKD", str(x))
            if not unicodedata.combining(c)
        )

    return s.map(_f).astype("string")


def normalize_player(s: pd.Series) -> pd.Series:
    """Normaliza nombre de jugador: limpia texto, mantiene acentos (nombre propio)."""
    return clean_text(s)
