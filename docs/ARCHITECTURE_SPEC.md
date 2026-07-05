# ARCHITECTURE SPEC — Mundial 2026 ML

Fuente de verdad de la arquitectura. Cualquier cambio de código que viole un
contrato de este documento debe actualizar primero este spec (Spec-Driven
Development).

Versión: 1.1 · 2026-06-09

---

## 1. Arquitectura Medallion (local, migrable a AWS)

```
Bronze (data/raw)      Silver (data/interim)       Gold (data/processed)
descargas inmutables → limpio, tipado, canónico  → features ML (1 fila = 1 partido)
        │                       ▲
        │                       │ consolidación (scripts/05)
        └──────────► Live (data/live) ── ingesta del torneo en curso
```

| Zona   | Ruta              | Formato | Mutabilidad | Equivalente AWS |
|--------|-------------------|---------|-------------|-----------------|
| Bronze | `data/raw/`       | CSV/JSON tal cual la fuente | INMUTABLE: solo la reescribe `scripts/00_download_tier0.py` | S3 bucket raw + Glue crawler |
| Silver | `data/interim/`   | Parquet | Solo la reescriben `scripts/01_build_silver.py` y `05_consolidate_live.py` (archivo NUEVO, nunca pisa los existentes) | S3 parquet particionado + Athena |
| Gold   | `data/processed/` | Parquet | Solo la reescribe `scripts/02_build_features.py` | S3 + Athena/feature store |
| Live   | `data/live/`      | CSV     | Append/delete SOLO vía `LiveStore` | DynamoDB o S3 + API Gateway |

**Invariante M1**: ningún componente de predicción escribe en Bronze/Silver/Gold.
**Invariante M2**: la consolidación (`LiveStore.consolidated_matches`) produce
`matches_consolidated.parquet` — un archivo nuevo; los originales no se tocan.
**Invariante M3**: todas las rutas derivan de `src/mundial/config.py`
(`PROJECT_ROOT` relativo al repo) o del `ROOT` de `app.py`. Prohibido
hardcodear rutas absolutas de máquina.

## 2. Contratos de datos

### 2.1 Silver: `matches.parquet`

| columna | tipo | contrato |
|---|---|---|
| date | datetime64 | orden cronológico estable (`kind="stable"`) |
| home_team / away_team | str | nombres canónicos (`transform/names.py`) |
| home_score / away_score | int | >= 0 |
| tournament | str | etiqueta de la fuente |
| neutral | bool | False solo si el local juega en su país |

### 2.2 Live: 4 CSV gestionados por `mundial.live.store.LiveStore`

- `live_results.csv` — `RESULT_COLS`: match_id, date, home_team, away_team,
  home_score, away_score, neutral, stage (`group`|`ko`), ko_winner (solo si
  hubo penales), xg_home, xg_away, weather, formation_home, formation_away.
  Campos de contexto son OPCIONALES (NA permitido).
- `live_players.csv` — `PLAYER_COLS`: match_id, date, team, player,
  event ∈ {goal, penalty, own_goal, assist}, minute.
- `live_discipline.csv` — `CARD_COLS`: match_id, date, team, player,
  card ∈ {yellow, red}, minute.
- `live_injuries.csv` — `INJURY_COLS`: match_id, date, team, player,
  severity ∈ {next_match, tournament}.

**Invariante L1**: `match_id = YYYYMMDD_home-slug_away-slug` une los 4 archivos;
`delete_match(match_id)` borra en cascada.
**Invariante L2**: `LiveStore.token()` cambia si cambia cualquiera de los 4
archivos → es la clave de invalidación de todos los cachés de Streamlit.
**Invariante L3**: lectura tolerante a esquemas viejos (columnas faltantes se
rellenan con NA); escritura siempre con el esquema completo.

### 2.3 Modelo: `models/artifacts.joblib`

dict con claves: `clf` (LogisticRegression calibrada, clases `['A','D','H']`),
`xgb` (XGBClassifier, mismas clases vía códigos 0/1/2), `pois_home`/`pois_away`
(pipelines Poisson), `rho` (Dixon-Coles, -0.175), `weights` (ensemble 3
modelos clf/xgb/pois, calibrado en validación: 0.70/0.00/0.30), `blend`
(compat 2-modelos), `n_train`, `trained_until`.

**Invariante F4 (valor de plantilla)**: features `home_log_value` /
`away_log_value` / `diff_log_value` = log10 del valor Transfermarkt de la
selección por año (script 07: top-26 valoraciones vigentes de sus
internacionales; fuente salimt/football-datasets). NaN sin cobertura
(pre-2005, selecciones chicas, Arabia Saudita/Jordania/Uzbekistán) → imputer.
El engine usa `_log_value(team, año)` con fallback de hasta 2 años.
**Invariante F5 (peso del XGB)**: el XGBoost solo participa si la calibración
le da peso > 0 (`engine.xgb_active`); con peso 0 no se paga su inferencia.
Evidencia 2026-06-10: val log-loss XGB 0.84 vs logística 0.83 — las features
actuales son monótonas y el boosting no aporta; re-evaluar cuando entren
features de interacción (fatiga×edad, viaje×descanso).

**Invariante F1**: `PredictionEngine.features_for` replica EXACTAMENTE las
features batch de `features/build.py`. Si cambias una, cambia ambos lados y
reentrena. **Invariante F2**: Elo base con K ponderado por torneo
(`elo.k_for`: Mundial 60, continentales 50, eliminatorias/Nations League 40,
amistosos 20, menores 30 — estándar World Football Elo); el replay del engine
usa el MISMO `k_for` que el entrenamiento, y la capa live aplica los partidos
del torneo con `tournament="FIFA World Cup"` (K=60). La capa live NUNCA
modifica `engine.elo` (el ruido `elo_sigma` del MC lo restaura con
`try/finally`). **Invariante F3**: `rest_days` se capa a 30
(`build.REST_DAYS_CAP`) en build Y en `features_for` — el histórico tiene gaps
de meses pero en torneo son 4-7 días (fuera de distribución sin el cap).

## 3. Flujo de State Updating + Online Learning

```
Ingesta UI ──► LiveStore (data/live/*.csv)
                  │  token() cambia
                  ▼
        LiveEngine.__init__ (reconstrucción cacheada)
          1. replay histórico (PredictionEngine base)
          2. por cada partido live EN ORDEN:
               a. match_distribution() con hooks APAGADOS  ──► OnlineCorrector.add_record
                  (predicción honesta pre-partido, sin leakage)
               b. TournamentState.record_match (momentum, con xG si existe)
               c. apply_result (Elo K=60 Mundial / forma / H2H base)
          3. TournamentState.load_context(tarjetas, lesiones)
          4. OnlineCorrector.fit()  ──► hooks ENCENDIDOS
                  │
                  ▼ predicción (cards, Eliminatorias, Monte Carlo)
          elo_for()        = elo base + momentum − sanciones/lesiones
          _adjust_lambdas() = λ × gamma_goles × factor_altitud(ciudad)
          _adjust_probs()   = P(empate) × draw_mult, renormalizado;
                              en eliminatorias además p^(1/ko_temp) renorm.
```

### Parámetros del estado (`live/state.py`)

| parámetro | valor | racional |
|---|---|---|
| K_LIVE | 45 | momentum extra del torneo = (K_LIVE−K)·g·(s−E) |
| XG_BLEND | 0.35 | margen efectivo = 0.65·goles + 0.35·xG |
| PEN_SUSPENSION | 9 Elo | jugador suspendido (roja, o 2 amarillas acumuladas) |
| PEN_INJ_MATCH / TOURN | 7 / 14 Elo | lesión un partido / resto del torneo |
| MAX_PENALTY | 45 Elo | tope por equipo |
| YELLOW_RESET | 2026-07-11 | FIFA limpia amarillas tras cuartos |

**Invariante S1**: una sanción/lesión `next_match` aplica solo hasta que el
equipo juegue su siguiente partido registrado.
**Invariante S2**: los ajustes son explicables — `explain(team, date)` devuelve
cada modificador con su etiqueta y puntos (contrato de XAI de la UI).

### Parámetros del corrector (`live/online.py`)

| factor | prior (shrinkage) | clip |
|---|---|---|
| gamma (ritmo de goles) | N0 = 20 goles esperados | [0.85, 1.18] |
| draw_mult (empates) | K0 = 12 partidos | [0.80, 1.55] |
| alt_mult (sedes ≥ 1400 m) | 1.05, N0 = 8 partidos | [0.90, 1.20] |
| ko_temp (afilado en KO) | 1.0, N0 = 15 partidos KO | [0.80, 1.10] |

**Invariante O1**: con 0 partidos, todos los factores = 1.0 (modelo base
intacto). **Invariante O2**: la capa live JAMÁS llama `model.fit()`.
**Invariante O3 (afilado KO, 2026-07-05)**: en el backtest del torneo los
favoritos de eliminatoria rinden por encima de su probabilidad declarada
(15/19 avanzaron; ll KO 0.690 → 0.625 con T=0.75). `ko_temp` se ajusta por
grid-search de log-loss SOLO sobre predicciones honestas pre-partido de
partidos KO ya jugados, con shrinkage hacia 1.0 (N0=15) y clip [0.80, 1.10];
se aplica `p^(1/T)` renormalizado únicamente a predicciones de eliminatoria
(fecha ≥ primer KO registrado), después de draw_mult y antes del mercado.
Sin partidos KO registrados, T=1.0 (O1 se conserva). Los partidos de grupos
JAMÁS se afilan (el óptimo medido en grupos es T=1.0).

**Invariante M5 (mercado)**: las cuotas 1X2 (`data/live/live_odds.csv`, tab
Mercado) se de-vig con `engine.devig()` (1/cuota normalizado) y se mezclan en
`match_distribution` tras los ajustes online y ANTES de reescalar la matriz:
`p = (1−w)·p_modelo + w·mercado`, `MARKET_WEIGHT=0.50`, SOLO donde hay cuota
para `(home,away)`. Sin cuota, comportamiento idéntico al modelo puro. El
`display_pred` de la UI (empate visible) es cosmético y NO altera el `pred`
que propaga el bracket (invariante U4).

## 4. Simulación Monte Carlo (`predict/montecarlo.py`)

- Grupos: marcadores muestreados de la matriz de cada partido (o el resultado
  real si está en live). Desempate FIFA: **Pts → DG → GF → head-to-head entre
  empatados (Pts/DG/GF del mini-grupo) → azar** (`rank_group`, función pura).
- Terceros: los 8 mejores por Pts → DG → GF (sin H2H entre grupos, regla FIFA).
- Eliminatorias: bracket oficial de openfootball; prórroga/penales con
  `tiebreak_prob` (logística Elo COMPRIMIDA hacia 0.5 con `TIEBREAK_DAMP=0.25`
  — los penales reales son ~50/50; la logística pura le daba 78% al favorito
  y concentraba P(campeón)); los cruces ya jugados (en live, con `ko_winner`
  si hubo penales) se respetan.
- **Incertidumbre de fuerza (M4)**: `run(elo_sigma=75)` perturba el Elo de los
  48 clasificados ~N(0, σ) por bloque de 250 sims con ruido ANTITÉTICO (el
  bloque impar niega el del par → media exacta 0 por equipo). El rating es una
  estimación, no una verdad: sin esto la ventaja del favorito se compone en
  7 rondas (Argentina llegaba a 24% de P(campeón) vs ~9% del mercado).
  `elo_sigma=0` reproduce el comportamiento determinista (tests).
  `engine.elo` SIEMPRE queda restaurado tras `run()` (try/finally).
- `slot_stats`: top-3 candidatos por lado de cada llave + ganador (alimenta la
  pestaña Cuadro).

## 5. UI (`app.py`)

### Diseño visual
- **Tipografía**: Poppins (Google Fonts) en todo el sistema.
- **Tema dual**: CSS variables inyectadas (`PALETTES` en dark/light) + toggle en
  UI. Fondo aurora animado con 3 blobs en drift perpetuo.
- **Glassmorphism**: `backdrop-filter: blur(28px) saturate(160%)` en cards,
  tabs, modales y métricas — consistente en ambos temas.
- **Motion UI**: transiciones `cubic-bezier(.16,1,.3,1)` en hover/active de
  botones, cards y métricas.
- **Scrollbar**: personalizada (6px, bordes redondeados).

### Componentes

| Componente | Implementación | Contrato |
|---|---|---|
| **match_card** | HTML inline con clases CSS (`glass`, `mc-*`). Muestra flags, nombres, y probabilidades tipográficas (H/D/A sin barra). Botón "Explicar pronóstico" por card. |  |
| **xai_dialog** | `st.dialog` con desglose completo: Elo base, momentum, sanciones/lesiones, ajuste total, Elo efectivo, correcciones online, λ Poisson, top-5 marcadores. | Invariante S2: `explain()` es la fuente de toda verdad XAI. |
| **tbl** | Tabla HTML propia con theming vía `tblwrap`/`tbl` clases. Soporta flags y barras de progreso. | Reemplaza a `st.dataframe` para respetar el tema oscuro/claro. |
| **leader-card** | Tarjetas animadas con gradiente, hover `translateY(-4px) scale(1.02)` y glow. | Top-10 goleadores, asistencias y tarjetas. |
| **form-card** | Contenedor glass con padding y hover sutil para los formularios de ingreso. | Separa visualmente las secciones de registro. |
| **bracket** | Flexbox horizontal con overflow-x auto. 6 columnas (R32 → Campeón) + conectores SVG entre llaves. Cada llave muestra ocupantes + P de ocupar la llave. | Invariante U4 (determinismo): los entrantes a R32 son los ocupantes modales del Monte Carlo, pero a partir de ahí la propagación es DETERMINISTA — en cada llave avanza el equipo con `P(avanza) > 50%` según `engine.predict_match` (o el ganador real si el cruce ya se ingresó). Las marginales de `slot_stats` NO componen entre rondas y solo se usan para los `pct` mostrados. |

### Flujo de ingesta de resultados (Tab 2)
- Dos paneles lado a lado: fase de grupos (selector de fixture pendiente) y
  eliminatorias/manual (selectores libres).
- Tooltips en inputs de goles y xG.
- `detail_block` dentro de cada panel con contexto opcional: xG, goleadores,
  tarjetas, lesiones, clima, formación.
- Tabla de resultados ingresados con flag de equipo y botón de borrado.

### Invariantes
- **U1**: toda predicción mostrada pasa por `LiveEngine` (nunca por el engine
  base directamente).
- **U2**: tras guardar/borrar un partido, el `token()` del store invalida
  `build_engine` y `run_simulation` — la UI nunca muestra estado viejo.
- **U3**: el modal XAI usa `engine.state.explain()` como única fuente — los
  valores en el modal y en la card son coherentes.
- **U5 (responsividad móvil)**: la página NUNCA tiene scroll horizontal a
  nivel de viewport. Estrategia mobile-first-fix con un único bloque
  `@media (max-width: 768px)` al final de `BASE_CSS` (desktop intacto:
  ninguna regla fuera del media query cambia). Los únicos contenedores con
  scroll-x propio son: la tab-list (pills, scrollbar oculta), `.tblwrap`
  (tablas) y el `#tree` del bracket DENTRO de su iframe
  (`-webkit-overflow-scrolling: touch`) — el swipe horizontal ocurre solo
  dentro del cuadro, el resto de la app queda anclada al viewport.
  Anchos absolutos prohibidos en móvil: todo `min-width` fijo de filas
  flex (`.audit-row`) se anula y la fila hace wrap.

## 6. Testing

- `tests/` con pytest; `pyproject.toml` define `pythonpath = ["src"]`.
- Cobertura mínima exigida: lógica de sanciones/lesiones (S1, reset de
  amarillas), consolidación (M2, dedup), desempate H2H (sección 4).
- Los tests no tocan `data/` real: usan `tmp_path`.

## 7. Frontend SPA inyectado (`src/frontend/`)

Los efectos web premium viven FUERA de `app.py` en `src/frontend/`
(templates HTML/JS/CSS) y se inyectan vía `st.components.v1.html` (iframe
same-origin que opera sobre `window.parent`). Degradables: si un CDN falla,
la app sigue 100% funcional.

### Contrato JSON del bracket (`render_bracket`)

```json
{
  "rounds": [{"key": "R32", "label": "Dieciseisavos",
              "matches": [{"num": 74, "src1": null, "src2": null,
                           "t1": {"team": "...", "flag": "url", "pct": 64},
                           "t2": {...}, "win": "team|null", "pwin": 62,
                           "cands1": [["team", 0.64], ...],
                           "date": "28 JUN", "ground": "..."}]}],
  "champion": {"team": "...", "flag": "url", "pct": 25}
}
```

- `num`: id de la llave (`"final"` para la final, sin num en el JSON fuente).
- `src1`/`src2`: num de la llave previa que alimenta cada lado (null en R32)
  — el template dibuja los conectores SVG con este mapeo, nunca por posición.
- `pwin`: P(%) de que `win` avance en ESTE cruce concreto (determinismo U4).
- `pct` puede ser null (cruce sin resolver); el template omite el
  porcentaje en ese caso.
- **Semántica de `pct` (U4-display)**: en cada llave, `t1.pct`/`t2.pct` es
  la probabilidad de AVANZAR EN ESE CRUCE concreto (suman ~100), NO la
  probabilidad marginal de ocupar la llave. Mostrar la ocupación marginal
  junto al ganador del cruce confunde (un equipo puede ser menos frecuente
  en el slot y aun así ser favorito del head-to-head). La ocupación
  marginal vive solo en `cands1/cands2` (modo foco, "Alt:").

### Contrato de theming (frontend)

- Única fuente de verdad: CSS variables `--bg/--surface/--text/--accent/...`
  inyectadas en el `:root` del documento padre por `inject_theme()`.
- PROHIBIDO hardcodear colores en JS/HTML inyectado: los componentes leen
  `getComputedStyle(parent.documentElement)` y se re-aplican con un
  **MutationObserver** sobre `<head>` (el toggle de tema re-inyecta el
  `<style>` de `:root`).
- Stack: Lenis (scroll), GSAP (transiciones de tab y bracket), Three.js
  (fondo liquid gradient con shader, colores = vars del tema).

## 8. RBAC (`src/mundial/auth.py`)

| Rol | Acceso | Activación |
|---|---|---|
| `viewer` (default) | Próximos, Líderes, Cuadro, Eliminatorias, Camino, Tablas, XAI | ninguna |
| `admin` | todo + **Ingresar resultado** (motor en vivo) | contraseña de `st.secrets["admin_password"]` (top-level, igual en local y en Streamlit Community Cloud; `[auth].admin_password` aceptado por compatibilidad) |

- La contraseña vive en `.streamlit/secrets.toml` (GITIGNORED); se versiona
  `secrets.toml.example`. Sin secrets configurados, la app queda en viewer.
- Invariante R1: el rol vive en `st.session_state["role"]`; el tab de
  ingesta NO se construye para viewers (no solo se oculta).
- UI de login: botón "Acceso admin" en el header → `st.dialog` modal
  (`auth.login_entry`). NO usa `st.sidebar`: el header nativo de Streamlit
  está oculto por CSS y se llevaba consigo el control de re-expandir el
  sidebar (bug conocido).
- Invariante R2 (no-leak): el flujo de login NUNCA muestra en pantalla
  detalles de configuración (rutas de secrets, nombres de claves,
  plantillas TOML) ni distingue entre "clave incorrecta" y "auth sin
  configurar" — siempre el mismo "Acceso denegado." genérico. Las
  instrucciones de despliegue viven en docs/CLAUDE.md, no en la UI.
- Invariante S3 (sanitización en el boundary de persistencia):
  `LiveStore.add_match` sanitiza todo texto libre (jugadores, formaciones)
  antes de escribir a CSV — sin caracteres de control, sin prefijos de
  fórmula (`= + - @`, anti CSV-injection en Excel/Sheets), longitud
  acotada. El escape HTML en display (`esc()`) se mantiene como segunda
  capa.

## 9. Tab "Auditoría" (backtesting visual)

- Selector de selección → últimos 5 partidos JUGADOS desde la capa Gold
  (`features.parquet`, features pre-partido con anti-leakage verificado).
- Para cada partido se reconstruye la predicción pre-partido con los
  artefactos entrenados (mismo ensemble blend·clf + (1-blend)·Poisson DC
  que producción) — NUNCA con el Elo actual del engine (sería leakage).
- Card por partido: fecha, rival, P(victoria) del equipo elegido,
  marcador real y ✓/✗ si el argmax 1X2 del modelo coincidió con el
  resultado. Tematizado con las CSS vars (dark/light).

### 2.4 Gold: `rosters_2026.parquet` (plantillas normalizadas)

| columna | tipo | contrato |
|---|---|---|
| team | str | nombre canónico (uno de los 64 en fixtures) |
| player | str | nombre normalizado (fuente: goalscorers histórico) |
| goals | int | goles registrados desde `since` (orden del dropdown) |
| last_seen | date | último gol registrado |

Generado por `scripts/06_build_rosters.py` (lógica testeable en
`mundial.ingest.rosters.build_rosters`). Los selectores de jugador de la UI
leen SOLO de aquí (anti-typos), con opción de escape "Otro…" para texto libre.
