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
    # plantilla Transfermarkt (script 07, NaN sin cobertura): valor log10,
    # edad ponderada por valor (no monótona: terreno del XGB) y
    # concentración del valor en el top-3 (dependencia de estrellas)
    "home_log_value", "away_log_value", "diff_log_value",
    "home_squad_age", "away_squad_age", "diff_top3_share",
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


def make_xgb() -> Pipeline:
    """Gradient boosting conservador (anti-overfit) para no linealidades.
    XGBoost maneja NaN nativamente — sin imputer, los valores de plantilla
    faltantes son informativos (selecciones sin cobertura de mercado)."""
    from xgboost import XGBClassifier

    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, objective="multi:softprob",
            eval_metric="mlogloss", tree_method="hist",
            n_jobs=-1, random_state=42)),
    ])


def fit_xgb(pipe: Pipeline, X, y) -> Pipeline:
    """Entrena el XGB con labels 'A'/'D'/'H' codificadas 0/1/2 (orden
    alfabético = el mismo de LogisticRegression.classes_)."""
    import pandas as pd

    codes = pd.Categorical(y, categories=LABELS_SORTED).codes
    return pipe.fit(X, codes)


LABELS_SORTED = ["A", "D", "H"]   # orden de clases de sklearn (alfabético)
