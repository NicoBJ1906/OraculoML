"""Sprint 5 - Modelo final para el Mundial 2026.

1. Evalúa honestamente (split temporal): clasificador vs Poisson vs ensemble.
2. Calibra rho (Dixon-Coles) y el peso del ensemble en validación 2020-2021.
3. Reentrena con TODO el histórico (2010 -> hoy) y guarda artefactos.

Uso:
    python scripts/04_train_final.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

from mundial.config import PROCESSED, PROJECT_ROOT  # noqa: E402
from mundial.models.baseline import (  # noqa: E402
    FEATURES, fit_xgb, make_logreg, make_xgb,
)
from mundial.models.poisson import (  # noqa: E402
    POISSON_FEATURES, estimate_rho, fit_goal_models, predict_proba_1x2,
)

MODELS_DIR = PROJECT_ROOT / "models"
LABELS_SORTED = ["A", "D", "H"]


def calibrate_weights(y_val, p_clf, p_xgb, p_pois,
                      step: float = 0.05) -> tuple[float, float, float]:
    """Pesos (clf, xgb, pois) que minimizan log-loss en validación,
    búsqueda en la grilla del simplex."""
    best, best_ll = (1.0, 0.0, 0.0), np.inf
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w3 = max(w3, 0.0)
            ll = log_loss(y_val, w1 * p_clf + w2 * p_xgb + w3 * p_pois,
                          labels=LABELS_SORTED)
            if ll < best_ll:
                best, best_ll = (float(w1), float(w2), float(w3)), ll
    return best


def report(name: str, y, proba) -> None:
    pred = np.array(LABELS_SORTED)[proba.argmax(axis=1)]
    print(f"  {name:24} acc={accuracy_score(y, pred):.3f}   "
          f"log-loss={log_loss(y, proba, labels=LABELS_SORTED):.3f}")


def main() -> None:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df = df[(df.year >= 2010)
            & (df.home_matches_prior >= 5)
            & (df.away_matches_prior >= 5)].copy()
    df["neutral"] = df["neutral"].astype(int)

    fit = df[df.year <= 2019]          # núcleo de entrenamiento
    val = df[df.year.between(2020, 2021)]   # calibra rho y blend
    test = df[df.year >= 2022]         # métrica honesta

    # --- entrenar en fit, calibrar en val ---
    clf = make_logreg().fit(fit[FEATURES], fit.result)
    xgb = fit_xgb(make_xgb(), fit[FEATURES], fit.result)
    ph, pa = fit_goal_models(fit[POISSON_FEATURES], fit.home_score, fit.away_score)
    rho = estimate_rho(ph, pa, val[POISSON_FEATURES], val.result)

    p_clf_val = clf.predict_proba(val[FEATURES])
    p_xgb_val = xgb.predict_proba(val[FEATURES])
    p_poi_val = predict_proba_1x2(ph, pa, val[POISSON_FEATURES], rho)
    weights = calibrate_weights(val.result, p_clf_val, p_xgb_val, p_poi_val)
    blend = weights[0] + weights[1]   # compat 2-modelos (clf+xgb vs pois)
    print(f"calibración: rho={rho:+.3f}   "
          f"weights(clf,xgb,pois)=({weights[0]:.2f}, {weights[1]:.2f}, "
          f"{weights[2]:.2f})\n")

    # --- evaluación honesta en test >= 2022 (modelos reentrenados <= 2021) ---
    tr = df[df.year <= 2021]
    clf_t = make_logreg().fit(tr[FEATURES], tr.result)
    xgb_t = fit_xgb(make_xgb(), tr[FEATURES], tr.result)
    ph_t, pa_t = fit_goal_models(tr[POISSON_FEATURES], tr.home_score, tr.away_score)
    p_clf = clf_t.predict_proba(test[FEATURES])
    p_xgb = xgb_t.predict_proba(test[FEATURES])
    p_poi = predict_proba_1x2(ph_t, pa_t, test[POISSON_FEATURES], rho)
    w1, w2, w3 = weights
    p_mix = w1 * p_clf + w2 * p_xgb + w3 * p_poi

    print(f"== Test temporal >= 2022 ({len(test)} partidos) ==")
    report("Logistic (clf)", test.result, p_clf)
    report("XGBoost", test.result, p_xgb)
    report("Poisson Dixon-Coles", test.result, p_poi)
    report("Ensemble (3 modelos)", test.result, p_mix)

    wc = test[(test.tournament == "FIFA World Cup") & (test.year == 2022)]
    if len(wc):
        p_wc = (w1 * clf_t.predict_proba(wc[FEATURES])
                + w2 * xgb_t.predict_proba(wc[FEATURES])
                + w3 * predict_proba_1x2(ph_t, pa_t, wc[POISSON_FEATURES], rho))
        pred_wc = np.array(LABELS_SORTED)[p_wc.argmax(axis=1)]
        print(f"\nHold-out Mundial 2022 ({len(wc)}): "
              f"acc ensemble = {accuracy_score(wc.result, pred_wc):.3f}   "
              f"log-loss = {log_loss(wc.result, p_wc, labels=LABELS_SORTED):.3f}")

    # --- modelo FINAL: reentrena con todo (2010 -> hoy) ---
    clf_f = make_logreg().fit(df[FEATURES], df.result)
    xgb_f = fit_xgb(make_xgb(), df[FEATURES], df.result)
    ph_f, pa_f = fit_goal_models(df[POISSON_FEATURES], df.home_score, df.away_score)

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / "artifacts.joblib"
    joblib.dump({
        "clf": clf_f, "xgb": xgb_f, "pois_home": ph_f, "pois_away": pa_f,
        "rho": rho, "blend": blend, "weights": weights,
        "trained_until": str(df.date.max().date()),
        "n_train": len(df),
    }, out)
    print(f"\nArtefactos finales -> {out}  "
          f"(entrenado con {len(df)} partidos hasta {df.date.max().date()})")


if __name__ == "__main__":
    main()
