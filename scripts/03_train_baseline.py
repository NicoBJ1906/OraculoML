"""Sprint 1 - Baseline: clasificación 1X2 con validación temporal.

Compara: baseline trivial vs. Elo puro vs. Logistic Regression vs. Random Forest.

Uso:
    python scripts/03_train_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss  # noqa: E402

from mundial.config import PROCESSED  # noqa: E402
from mundial.models.baseline import (  # noqa: E402
    FEATURES, LABELS, make_logreg, make_rf, temporal_split,
)


def report(name: str, y_true, y_pred, proba, labels) -> None:
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, proba, labels=labels)
    print(f"  {name:26} acc={acc:.3f}   log-loss={ll:.3f}")


def main() -> None:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df = df[(df.year >= 2010)
            & (df.home_matches_prior >= 5)
            & (df.away_matches_prior >= 5)].copy()
    df["neutral"] = df["neutral"].astype(int)

    train, test = temporal_split(df, train_max_year=2021)
    Xtr, ytr = train[FEATURES], train["result"]
    Xte, yte = test[FEATURES], test["result"]
    print(f"train={len(train)} (<=2021)   test={len(test)} (>=2022)\n")

    print("== Resultados (test temporal >= 2022) ==")
    # 1) baseline trivial: siempre gana local
    maj = ytr.mode()[0]
    freq = ytr.value_counts(normalize=True)
    sorted_labels = sorted(LABELS)  # ['A','D','H'] como espera sklearn
    proba_triv = np.tile([freq.get(lab, 0.0) for lab in sorted_labels], (len(yte), 1))
    report("baseline (siempre local)", yte, [maj] * len(yte), proba_triv, sorted_labels)

    # 2) Elo puro (logistic con una sola feature)
    elo = make_logreg().fit(Xtr[["elo_diff"]], ytr)
    report("solo Elo (logistic)", yte, elo.predict(Xte[["elo_diff"]]),
           elo.predict_proba(Xte[["elo_diff"]]), elo.classes_)

    # 3) Logistic Regression (todas las features)
    lr = make_logreg().fit(Xtr, ytr)
    report("Logistic Regression", yte, lr.predict(Xte),
           lr.predict_proba(Xte), lr.classes_)

    # 4) Random Forest
    rf = make_rf().fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    report("Random Forest", yte, rf_pred, rf.predict_proba(Xte), rf.classes_)

    # ---- detalle del mejor modelo (Logistic) ----
    print("\nMatriz de confusión (Logistic) filas=real, cols=pred, orden",
          list(lr.classes_))
    print(confusion_matrix(yte, lr.predict(Xte), labels=lr.classes_))

    importances = pd.Series(
        rf.named_steps["clf"].feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print("\nTop 8 features (Random Forest):")
    print(importances.head(8).round(3).to_string())

    # ---- hold-out específico: Mundial 2022 ----
    wc = test[(test.tournament == "FIFA World Cup") & (test.year == 2022)]
    if len(wc):
        acc_wc = accuracy_score(wc.result, lr.predict(wc[FEATURES]))
        print(f"\nHold-out Mundial 2022 ({len(wc)} partidos): "
              f"acc Logistic = {acc_wc:.3f}")


if __name__ == "__main__":
    main()
