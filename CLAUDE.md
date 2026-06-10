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

## Sesión 2026-06-09 (noche) — producción y publicación

- **SDD**: `docs/ARCHITECTURE_SPEC.md` es la fuente de verdad (contratos e
  invariantes S1/S2, L1/L2, M2, O1/O2). Cambios de código que los violen
  deben actualizar el spec primero.
- **Tests**: 19 pytest en verde (`tests/`): sanciones/lesiones/reset de
  amarillas, consolidación con dedup, desempate H2H. `pyproject.toml` define
  `pythonpath=["src"]`. Correr con `.venv/Scripts/python -m pytest`.
- **XAI**: `TournamentState.explain(team, date)` → momentum + items con
  etiqueta y puntos. UI: chip "Δ EN VIVO" en cards, expander de desglose en
  Próximos, detalle por equipo en Eliminatorias.
- **Editores de captura**: `dynamic_rows()` con widgets DOM (st.data_editor
  canvas eliminado — no se podía tematizar). Al guardar se limpian via
  `rows_{ev,cd,in}_{prefix}` en session_state.
- **Monte Carlo**: `rank_group()` (función pura) aplica desempate FIFA
  completo Pts→DG→GF→H2H entre empatados→azar.
- **GitHub**: repo privado `NicoBJ1906/mundial-2026-ml`, rama main. `gh` NO
  está instalado: usar API REST con `$GITHUB_TOKEN` y push con
  `git -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"`
  (nunca persistir el token en .git/config).
- `.streamlit/config.toml` ahora en paleta roja (#ff2d55) para que los
  widgets nativos coincidan con el tema dark.

## Sesión 2026-06-10 — frontend SPA, bracket, RBAC, rosters

- `src/frontend/`: effects.html (Lenis+GSAP+Three.js liquid gradient) y
  bracket.html (componente autocontenido: filtro de fases con GSAP elastic,
  modo foco con grid). Inyectados vía components.html (iframe same-origin).
- **Theming de iframes (CLAVE)**: el `<style>` de `:root` que inyecta
  st.markdown vive en el BODY del padre (no en head) y Streamlit lo muta
  como characterData. Los componentes parsean las vars del último <style>
  con `--bg:` y se re-aplican con MutationObserver sobre `PD.body`
  {childList, subtree, characterData} con debounce rAF. NO usar
  getComputedStyle solo (devuelve valores viejos en el rerun).
- Bracket viejo con margin-math eliminado (era el daño principal).
- RBAC (`mundial/auth.py`): viewer default, admin con clave de
  `.streamlit/secrets.toml` (gitignored — la clave NUNCA va en archivos
  versionados; verla en el propio secrets.toml local). El tab Ingresar
  no se construye para viewers.
- Rosters Gold: `scripts/06_build_rosters.py` → rosters_2026.parquet
  (825 jugadores desde goalscorers 2022+). Dropdowns anti-typos con "Otro…".
- XAI dialog: sección pedagógica de Elo con st.latex.
- 23 tests verdes. Push: usar el extraHeader Basic documentado arriba.

## Sesión 2026-06-10 (tarde) — hotfix UI + determinismo del Cuadro

- **Login sin sidebar**: `auth.login_entry()` (botón en header → `st.dialog`).
  Causa raíz: el CSS oculta `header[data-testid="stHeader"]` y se llevaba el
  control de re-expandir el sidebar. El sidebar ya no se usa; el CSS deja
  visible `stSidebarCollapsedControl` por si algo vuelve a renderizar ahí.
- **Cuadro determinista (invariante U4 del spec)**: `build_bracket_payload`
  (cacheada, app.py) — entrantes a R32 = ocupante modal del Monte Carlo; de
  ahí en adelante avanza el de `p_advances > 50%` vía `engine.predict_match`
  (o el ganador real ingresado). Las marginales de `slot_stats` NO componen
  entre rondas; solo se usan para los pct. Payload ganó `num/src1/src2/pwin`.
- **Conectores del bracket**: SVG overlay en bracket.html, mapeado por
  `src1/src2` (num de llave), NUNCA por posición (el orden visual no
  coincide con los cruces, ej. llave 89 = W74 vs W77).
- **Recuadro negro del bracket en modo claro**: causa = `color-scheme` del
  iframe distinto al del embedder → el browser fuerza canvas opaco.
  Fix: `applyTheme()` copia el `colorScheme` computado del body padre.
- **Filtro de Próximos**: jornadas derivadas del calendario real (el n-ésimo
  partido de cada equipo es su Fecha n), con rango de fechas en el label.
- **Podio**: clases `.podium-pct`/`.podium-lbl` (la vieja `mc-score` no
  existía en el CSS — por eso "34%campeón" salía pegado).
- Smoke test sin Playwright: `python -c "import app"` (bare mode) +
  validación de consistencia del payload. 23 tests verdes.
- **Logging**: `logs/app.log` (RotatingFileHandler, logger raíz "mundial").
  Registra: carga de artifacts, build del engine, Monte Carlo (n y tiempo),
  bracket determinista, login OK/fallido (nunca la clave), guardar/borrar
  resultados. logs/ está gitignored.
- **Validación E2E con Playwright (2026-06-10)**: tabs, login modal → tab
  Ingresar aparece, jornadas reales (Fecha 1 = 24 partidos 11-17 jun),
  bracket determinista verificado en UI (Colombia 37% vs Croatia 45% de
  ocupar la llave → "Avanza Colombia · 65% en este cruce"), modo claro sin
  recuadro negro, conectores visibles, podio espaciado, guardar+borrar
  resultado con recálculo. 0 errores de consola.
- Fix tabla "Resultados ingresados": columnas renombradas a nombres ÚNICOS
  (GL/GV, xG (L)/(V)) — con nombres duplicados `r[c]` devuelve una Series
  y la celda imprimía "Name: 0, dtype: object".
- OJO si la app "pierde" funciones nuevas de módulos de src/ tras editar:
  reiniciar streamlit — el proceso viejo mantiene los módulos importados
  en caché (el AttributeError de login_entry fue eso, no un bug).

## Sesión 2026-06-10 (cierre) — data lake al día + secrets para Cloud

- **Pipeline corrido completo** (00→01→02→04→05→06): datos hasta
  2026-06-09, 49.398 partidos silver (+22), modelo reentrenado con 15.618
  (acc 0.602 / log-loss 0.867 / WC2022 hold-out 0.547 — estable), rosters
  regenerados. El Elo base ya incluye los amistosos de la víspera.
- **Secrets formato Cloud**: `st.secrets["admin_password"]` top-level
  (mismo formato local y en Community Cloud → Settings → Secrets);
  `[auth].admin_password` sigue aceptado. La clave local vive SOLO en
  `.streamlit/secrets.toml`. El modal admin muestra un st.info temporal
  con las instrucciones de despliegue (TODO: quitarlo al abrir al público).
- **CSS**: date picker (st.date_input, tab Eliminatorias) y modal de login
  variabilizados — cero colores quemados, funcionan en dark y light. Se
  usan las vars propias (`--text/--bg/...`) y NO las nativas de Streamlit
  (`--text-color`): las nativas vienen de config.toml y no cambian con el
  toggle runtime de tema.
- OJO: Streamlit avisa que `st.components.v1.html` se elimina después de
  2026-06-01 (fecha ya vencida) — migrar a `st.iframe` pronto.

## Pendiente / ideas

- `tests/` y `evaluate/` siguen vacíos (los chequeos viven en los scripts).
- Sprints 3-4 del PLAN (API-Football, StatsBomb xG) sin empezar.
- Migrar `components.html` → `st.iframe` (deprecación vencida, ver arriba).
- Expectativa del usuario: "predicciones altas" — ya se le explicó que
  55-60% en 1X2 es el techo honesto; el valor real está en la calibración.
