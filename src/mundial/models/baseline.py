"""Modelos baseline de clasificación 1X2 con validación temporal."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "diff_form_pts", "diff_form_gf", "diff_form_ga",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "home_rest_days", "away_rest_days", "diff_rest_days",
    "neutral",
]
LABELS = ["H", "D", "A"]


def temporal_split(df, train_max_year: int = 2021):
    """Split TEMPORAL (no aleatorio) para evitar leakage."""
    train = df[df.year <= train_max_year]
    test = df[df.year > train_max_year]
    return train, test


def make_logreg() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=42)),
    ])
