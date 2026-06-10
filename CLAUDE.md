# Mundial 2026 ML — Estado del proyecto

Contexto para Claude Code. Última sesión de trabajo: 2026-06-09 (dos días antes
del arranque del Mundial, 11/jun/2026). Refactor mayor: capa live completa
(Feature State Updating + Online Learning) y UI con temas.

## Qué es

Predictor del Mundial 2026: ensemble **Logistic Regression (0.8) + Poisson
Dixon-Coles (0.2, rho=-0.15)** sobre data lake medallion local
(raw → interim → processed → live). UI Streamlit con temas dark (negro+rojo,
aurora animada) y light minimalista, ingesta en vivo extendida (xG,
goleadores, asistencias, tarjetas, lesiones, clima, formación) y
reevaluación dinámica sin reentrenar.

## Cómo se usa

```bash
# UI (lo único necesario durante el Mundial)
.venv/Scripts/python.exe -m streamlit run app.py

# refrescar datos + reentrenar (opcional, antes del torneo o entre fases)
python scripts/00_download_tier0.py   # martj42 se actualiza a diario
python scripts/01_build_silver.py
python scripts/02_build_features.py
python scripts/04_train_final.py      # guarda models/artifacts.joblib
python scripts/05_consolidate_live.py # histórico+live -> matches_consolidated.parquet
```

Durante el torneo: pestaña **Ingresar resultado** → `LiveEngine`
(`src/mundial/live/engine.py`) actualiza Elo/forma/H2H + momentum/sanciones
/lesiones + corrección online y recalcula todo (cards, bracket, Monte Carlo).

## Capa live (src/mundial/live/) — datos en data/live/

- `store.py`  : 4 CSV (live_results con xG/clima/formación/ko_winner,
  live_players, live_discipline, live_injuries). `delete_match()` borra en
  cascada. `token()` invalida cachés de Streamlit. raw/interim nunca se pisan.
- `state.py`  : ajustes Elo SOLO en predicción — momentum (K_LIVE=45 extra
  sobre el K=30 base, margen efectivo goles/xG), suspensiones FIFA (roja o
  2 amarillas => 1 partido; amarillas se limpian tras cuartos), lesiones
  (un partido / resto del torneo). Heurísticas: -9/-7/-14 Elo, tope -45.
- `online.py` : corrección con shrinkage bayesiano, factores=1.0 con 0
  partidos — gamma de goles (prior 20 xG), multiplicador de empates (prior
  25 partidos), altitud >=1400m (CDMX/Guadalajara, prior 1.05). Usa xG
  ingresado (blend 0.65 goles / 0.35 xG). NUNCA hace model.fit().
- `engine.py` : LiveEngine(PredictionEngine) — replay live registrando
  predicciones pre-partido honestas para el corrector (hooks apagados
  durante el replay, sin leakage).

`PredictionEngine` ganó 3 hooks no-op (elo_for, _adjust_lambdas,
_adjust_probs) — el engine base sigue siendo bit-a-bit el de entrenamiento.

## Estado y métricas (validación temporal, test 2022+, n=4519)

- Acierto 1X2: **0.602** | log-loss 0.867 (baseline local: 0.477 / 1.051)
- Marcador exacto de la card: **13.1%** (referencia siempre-1-0: 10.8%)
- **Calibración casi perfecta** verificada por buckets (dice 80% → acierta 80%)
- Torneos grandes (WC/Euro/Copa América): ~53% — techo del estado del arte
- Hold-out Mundial 2022: 54.7%
- Datos hasta 2026-06-08; los 72 fixtures de grupos vienen en results.csv con
  scores NA y flag `neutral` correcto (anfitriones USA/MEX/CAN = False)

## Decisiones técnicas clave (no romper)

- `PredictionEngine` replica EXACTAMENTE las features batch de
  `features/build.py` (verificado numéricamente). Si cambias una feature,
  cambia ambos lados. Sorts con `kind="stable"` para determinismo.
- `predict_match` reescala la matriz de marcadores para que sus marginales 1X2
  coincidan con el ensemble (coherencia card/porcentajes) y expone
  `score_pred` = marcador más probable DEL resultado predicho (lo que muestra
  la card). Sin esto, 35/72 cards contradecían sus propios porcentajes.
- Corrección Dixon-Coles con clip a 0 (rho grande puede dar prob. negativas).
- Elo: K=30 plano (consistente con el entrenamiento; no subir a 60 en WC sin
  reentrenar todo).
- Banderas vía **flagcdn.com** (`FLAG_ISO` en app.py): los emojis de bandera
  NO renderizan en Windows. flagcdn solo tiene anchos w40/w80/w160/w320
  (w60 da 404 — `flag_img` mapea al más cercano).
- UI 2026-06-09: temas light/dark con CSS variables (`PALETTES` +
  `BASE_CSS`), toggle "Modo claro" en runtime, aurora animada. Las tablas
  de display son HTML propio (`tbl()`) porque st.dataframe no se puede
  tematizar en runtime. Los st.data_editor del formulario (canvas) siguen
  el tema de `.streamlit/config.toml` (dark) — limitación conocida en modo
  claro, solo afecta el formulario.
- Tab Cuadro: bracket con ocupante más probable por llave; el simulador
  trackea `slot_stats` (top-3 candidatos por lado + ganador por llave) y
  respeta resultados KO ya ingresados (`ko_actual` por par de equipos,
  con `ko_winner` para empates resueltos por penales).
- Empates: el modelo les asigna la probabilidad correcta (23% medio = 23%
  real) pero casi nunca son argmax — comportamiento esperado, no es bug.

## Datos: qué hay y qué NO hay (auditado 2026-06-09)

- HAY: resultados 1872→hoy (49.450), goleadores históricos con minuto/
  penal/autogol (47.601, hasta mar-2026, SIN asistencias), shootouts,
  fixtures y bracket 2026, estadios (sin altitud — hardcodeada en
  `online.py:ALTITUDE_M`).
- NO HAY histórico de: xG, tarjetas, lesiones, clima, alineaciones. Por
  eso NO son features entrenadas: entran como ajustes en vivo (state +
  online). Si algún día se integra StatsBomb/API-Football, ahí sí podrían
  ser features y habría que reentrenar.

## Pendiente / ideas

- `tests/` y `evaluate/` siguen vacíos (los chequeos viven en los scripts).
- Sprints 3-4 del PLAN (API-Football, StatsBomb xG) sin empezar.
- Cruces de eliminatoria: el bracket ya se autocompleta vía Monte Carlo;
  falta resolver determinísticamente los cruces R32 cuando los grupos
  cierren (hoy quedan implícitos en las frecuencias del sim).
- Expectativa del usuario: "predicciones altas" — ya se le explicó que
  55-60% en 1X2 es el techo honesto; el valor real está en la calibración.
