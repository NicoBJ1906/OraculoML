# Plan robusto y escalonado — Mundial 2026 ML

## 1. Objetivo

Aprender **regresión** y **clasificación** construyendo dos modelos profesionales
sobre un data lake de fútbol de selecciones (2010–2026), con rigor de ingeniería:
arquitectura por capas, validación temporal, control de *data leakage* y baselines.

Entrenamiento con partidos **pasados y conocidos** (2010 → jun 2026). El Mundial
2026 (arranca 11/06/2026) es la **demo final**: se le pasan los fixtures al modelo
ya entrenado. No se entrena con datos que aún no existen.

## 2. Principio rector: nada de leakage

- **Features solo miran al pasado** del partido que predicen ("últimos 5", H2H,
  acumulados, Elo se calculan con fecha de corte = día del partido).
- **Split temporal, no aleatorio**: train ≤ 2021, validación hold-out = Mundial
  2022. K-fold aleatorio queda prohibido (inflaría la métrica).
- Tests automáticos en `tests/` que verifican que ninguna feature use el resultado
  del propio partido.

## 3. Arquitectura (medallion local, lista para AWS)

```
Ingesta (Python)
   │  raw/  (bronze)   descargas tal cual: CSV/JSON
   ▼
 silver/  (interim)   limpio, tipado, deduplicado, Parquet
   ▼
 gold/   (processed)  features ML: 1 fila = 1 partido (home/away)
   ▼
 Entrenamiento local  scikit-learn / statsmodels / xgboost
```

Hoy todo en disco local. Mañana, mismo diseño → `raw=S3 bronze`, transformaciones
→ Athena SQL, sin reescribir la lógica. **No se toca cloud hasta tener modelos.**

## 4. Fuentes de datos — dos datasets, dos propósitos

| | Dataset AMPLIO | Dataset PROFUNDO |
|---|---|---|
| Cobertura | 2010–jun 2026, todas las selecciones | Mundiales 2018 + 2022 |
| Fuentes | martj42 + Elo propio + openfootball (+ API-Football opc.) | StatsBomb open-data |
| Features | resultado, goles, H2H, forma, acumulados, sedes | xG, presión, pases, recuperaciones altas |
| Rol | **entrena y predice el Mundial 2026** | sección estrella del informe (análisis avanzado) |

Comparten el esquema `gold` (una fila por partido). StatsBomb solo cubre Mundiales,
por eso va como estudio profundo aparte y no se mezcla en una tabla con 90% de NaN.

### Fuentes concretas (todas gratis, sin API key salvo aviso)
- **martj42/international_results** (GitHub): `results.csv`, `goalscorers.csv`,
  `shootouts.csv`, `former_names.csv`. Incluye **todas las eliminatorias**.
- **openfootball/worldcup.json** (GitHub): equipos, estadios (altitud), calendario 2026.
- **Elo**: se calcula desde los resultados (Sprint 2) — más educativo y sin dependencias.
- **StatsBomb open-data** (`statsbombpy`): event data Mundiales 2018/2022.
- **API-Football v3** (opcional, Sprint 3): posesión, tiros, córners, lineups.
  Límite free = 100 req/día → cachear todo en `raw/`.

## 5. Features por categoría → fuente (tu lista, reconciliada)

| Categoría | Features | Fuente | Sprint |
|---|---|---|---|
| Resultado / goles | local/visita, goles, 1X2 | martj42 | 1 |
| Head-to-Head (5 años) | victorias/empates H2H | martj42 (derivado) | 1 |
| Forma (últimos 5) | pts, goles a favor/contra | martj42 (derivado) | 1 |
| Fuerza | Elo pre-partido | calculado | 2 |
| Acumulados torneo | GF, GC, dif., porterías a 0 | martj42 (derivado) | 2 |
| Goleador | goles top scorer, dependencia % | goalscorers.csv | 2 |
| Contexto geográfico | altitud, km de viaje | openfootball + Haversine | 3 |
| Contexto torneo | fase, es_derbi | metadata + curado | 3 |
| Táctica/estilo | posesión, tiros, córners, faltas | API-Football | 3 |
| Plantilla | edad media, valor (proxy SoFIFA) | API-Football / SoFIFA | 3 |
| Dinámica avanzada | xG, presión, pases, recuperaciones | StatsBomb | 4 |

> Bajadas de prioridad: **lesiones históricas** (mal documentadas) y **valor de
> mercado** vía scraping (frágil → se usa rating FIFA/SoFIFA como proxy).

## 6. Los dos modelos

- **Clasificación — resultado 1·X·2.** Baseline → Logistic Regression →
  Random Forest → XGBoost. Métricas: accuracy, **log-loss**, **Brier**, matriz de
  confusión y **curva de calibración** (clave: las probabilidades deben ser honestas).
- **Regresión — goles con Poisson (Dixon-Coles).** Modela goles esperados de cada
  equipo; de la distribución conjunta se derivan P(1), P(X), P(2) y over/under.
- **Comparación final**: clasificación directa vs. probabilidades derivadas del
  Poisson. Esa comparación es el corazón académico del proyecto.

## 7. Evaluación y baselines (obligatorios)

1. "Siempre gana el local".
2. **Elo** (probabilidad por diferencia de rating).
3. (Opcional) cuotas de casas de apuestas, si se consiguen.

Expectativa realista y honesta: ~50–55% de acierto en 1X2 es el techo del estado
del arte; los empates son casi impredecibles. Igualar al Elo ya es buen resultado.

## 8. Roadmap por sprints (cada uno deja algo funcionando)

| Sprint | Entregable | Definición de "hecho" |
|---|---|---|
| **0** | Repo + venv + estructura | `scripts/00_download_tier0.py` corre sin error |
| **1** | Ingesta Tier-0 + silver + features base + baseline 1X2 | modelo end-to-end en local, métricas vs. baseline |
| **2** | Elo propio + Poisson + validación temporal + MLflow | dos modelos comparados, sin leakage (tests verdes) |
| **3** | Enriquecer AMPLIO (API-Football, geografía) + ablation | medir aporte real de cada bloque de features |
| **4** | Dataset PROFUNDO StatsBomb + análisis xG/espacial | features de élite + mapas de pases/tiros |
| **5** | Predicción Mundial 2026 partido a partido + informe | predicciones reales + documento final |

## 9. Cloud (futuro, fuera de alcance ahora)

Migración opcional: `raw → S3`, transformaciones → Athena SQL, ingesta → Lambda.
100% serverless/on-demand. Guardarraíles: billing alarm + AWS Budgets; **nunca**
crear RDS, EC2, NAT Gateway ni endpoints de SageMaker (de ahí vienen los cobros
accidentales). El entrenamiento sigue siendo local (coste cero).
