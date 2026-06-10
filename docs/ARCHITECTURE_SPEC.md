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
`pois_home`/`pois_away` (pipelines Poisson), `rho` (Dixon-Coles, -0.15),
`blend` (peso del clasificador, 0.8), `n_train`, `trained_until`.

**Invariante F1**: `PredictionEngine.features_for` replica EXACTAMENTE las
features batch de `features/build.py`. Si cambias una, cambia ambos lados y
reentrena. **Invariante F2**: Elo base con K=30 plano (igual que en
entrenamiento); la capa live NUNCA modifica `engine.elo`.

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
               c. apply_result (Elo K=30 / forma / H2H base)
          3. TournamentState.load_context(tarjetas, lesiones)
          4. OnlineCorrector.fit()  ──► hooks ENCENDIDOS
                  │
                  ▼ predicción (cards, Eliminatorias, Monte Carlo)
          elo_for()        = elo base + momentum − sanciones/lesiones
          _adjust_lambdas() = λ × gamma_goles × factor_altitud(ciudad)
          _adjust_probs()   = P(empate) × draw_mult, renormalizado
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
| draw_mult (empates) | K0 = 25 partidos | [0.80, 1.25] |
| alt_mult (sedes ≥ 1400 m) | 1.05, N0 = 8 partidos | [0.90, 1.20] |

**Invariante O1**: con 0 partidos, todos los factores = 1.0 (modelo base
intacto). **Invariante O2**: la capa live JAMÁS llama `model.fit()`.

## 4. Simulación Monte Carlo (`predict/montecarlo.py`)

- Grupos: marcadores muestreados de la matriz de cada partido (o el resultado
  real si está en live). Desempate FIFA: **Pts → DG → GF → head-to-head entre
  empatados (Pts/DG/GF del mini-grupo) → azar** (`rank_group`, función pura).
- Terceros: los 8 mejores por Pts → DG → GF (sin H2H entre grupos, regla FIFA).
- Eliminatorias: bracket oficial de openfootball; prórroga/penales por Elo;
  los cruces ya jugados (en live, con `ko_winner` si hubo penales) se respetan.
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
- `pct` puede ser null (ocupante determinista fuera del top-3 del Monte
  Carlo); el template omite el porcentaje en ese caso.

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
