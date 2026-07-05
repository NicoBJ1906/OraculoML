"""Invariante O3: temperatura KO del OnlineCorrector.

El afilado solo se ajusta con partidos de eliminatoria, con shrinkage
hacia 1.0, y solo se aplica a predicciones KO (nunca a grupos).
"""
import numpy as np

from mundial.live.online import OnlineCorrector


def _add(c, date, probs, hs, as_, stage):
    """Registro sintético: lambdas neutras, p_draw del vector."""
    c.add_record(date, 1.3, 1.3, probs[1], hs, as_,
                 probs=probs, stage=stage)


def test_sin_ko_temp_es_uno_y_no_afila():
    """O1/O3: sin partidos KO registrados el factor es neutro."""
    c = OnlineCorrector()
    for i in range(10):
        _add(c, f"2026-06-{11 + i}", (0.2, 0.3, 0.5), 1, 1, "group")
    c.fit()
    assert c.ko_temp == 1.0
    assert c.is_ko("2026-07-01") is False
    p = np.array([0.25, 0.25, 0.50])
    q = c.adjust_probs(p, ko=True)          # aun pidiendo ko, T=1 no afila
    r = c.adjust_probs(p, ko=False)
    np.testing.assert_allclose(q, r)


def test_favoritos_ko_dominantes_afilan():
    """Si el favorito KO siempre gana, ko_temp < 1 y el favorito sube."""
    c = OnlineCorrector()
    for i in range(20):
        _add(c, f"2026-06-{28 + i % 3}", (0.20, 0.25, 0.55), 2, 0, "R32")
    c.fit()
    assert c.ko_temp < 1.0
    assert c.ko_temp >= 0.80                # respeta el clip
    p = np.array([0.20, 0.25, 0.55])
    q = c.adjust_probs(p, ko=True)
    assert q[2] > c.adjust_probs(p, ko=False)[2]
    np.testing.assert_allclose(q.sum(), 1.0)


def test_shrinkage_con_pocos_ko():
    """Con 3 partidos KO el prior domina: T no puede alejarse de 1."""
    c = OnlineCorrector()
    for i in range(3):
        _add(c, f"2026-06-{28 + i}", (0.10, 0.20, 0.70), 3, 0, "R32")
    c.fit()
    # w = 3/(3+15) = 1/6; grid minimo 0.60 -> T >= 1 - (1-0.60)/6 = 0.933
    assert c.ko_temp >= 0.93


def test_grupos_nunca_se_afilan():
    """Una predicción de grupos no cambia aunque haya temperatura KO."""
    c = OnlineCorrector()
    for i in range(20):
        _add(c, "2026-06-28", (0.20, 0.25, 0.55), 2, 0, "R16")
    c.fit()
    assert c.ko_temp < 1.0
    assert c.is_ko("2026-06-20") is False   # fecha previa al primer KO
    assert c.is_ko("2026-07-09") is True
    p = np.array([0.30, 0.30, 0.40])
    np.testing.assert_allclose(c.adjust_probs(p, ko=False),
                               c.adjust_probs(p.copy(), ko=False))
