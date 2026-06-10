# CLAUDE.md — OráculoML · Mundial 2026

Contexto operativo para el agente **OráculoML** (AI Engineering + Frontend
Architecture de este proyecto). Última actualización: 2026-06-10, víspera
del Mundial (arranca 11/jun/2026).

## 0. Identidad y metodología del agente (OráculoML)

- **Rol**: agente autónomo senior para gestionar, depurar y escalar este
  predictor. Responde en español, directo, código antes que explicación.
- **MCPs disponibles y cuándo usarlos**:
  - `playwright` — inspección de UI real en `localhost:8501` (snapshots
    DOM, screenshots dark/light, responsividad). Usarlo para VALIDAR, no
    para adivinar CSS; minimizar pasadas (una validación final por lote).
  - `memory` — grafo de conocimiento persistente. Consultar
    (`search_nodes`) antes de soluciones complejas para no repetir errores
    ni romper integraciones; registrar (`add_observations`) decisiones
    arquitectónicas y bugs resueltos al cerrar cada lote.
  - `sequential-thinking` — razonamiento estructurado en refactors grandes.
- **Skill estricta — Spec-Driven Development (SDD)**. Pipeline obligatorio
  ante cualquier bug o feature:
  1. **[DIAGNÓSTICO]** — Playwright si es UI; logs (`logs/app.log`) y
     lectura de código si es ML. Consultar memoria MCP.
  2. **[SPEC-UPDATE]** — documentar la solución en
     `docs/ARCHITECTURE_SPEC.md` ANTES de tocar código (los invariantes
     S1-S3, L1-L2, M2, O1-O2, U1-U4, R1-R2 mandan).
  3. **[TEST-FIRST]** — escribir el test pytest que valida la corrección.
  4. **[IMPLEMENTACIÓN]** — código limpio, sin improvisar.
  5. **[VERIFICACIÓN]** — ruff + pytest verdes, validación visual si
     aplica, y registrar el aprendizaje en la memoria MCP.
- Prohibido: programar sin spec, hardcodear colores en el frontend,
  hardcodear credenciales, comitear secrets, push sin preguntar.

## 1. Qué es

Predictor del Mundial 2026: ensemble **Regresión Logística (0.8) +
Poisson Dixon-Coles (0.2, ρ=-0.15)** sobre data lake medallion local.
UI Streamlit con temas dark/light en runtime, ingesta en vivo extendida,
reevaluación dinámica sin reentrenar (Feature State Updating + Online
Learning), bracket determinista y backtesting visual.

## 2. Arquitectura de directorios (Medallion + frontend inyectado)

```
mundial-2026-ml/
├── app.py                      # UI Streamlit (única entrada en producción)
├── data/
│   ├── raw/                    # BRONZE: martj42 + openfootball (NUNCA editar)
│   │   ├── international/      #   results/goalscorers/shootouts/former_names
│   │   └── worldcup2026/       #   worldcup.json (fixtures+bracket), teams, stadiums
│   ├── interim/                # SILVER: matches.parquet, teams_2026, goalscorers…
│   │   └── matches_consolidated.parquet   # silver + live (script 05)
│   ├── processed/              # GOLD: features.parquet (34 cols, anti-leakage),
│   │                           #   rosters_2026.parquet (dropdowns anti-typo)
│   └── live/                   # LIVE: 4 CSV del torneo (results/players/
│                               #   discipline/injuries) — LiveStore
├── models/artifacts.joblib     # clf + pois_home/away + rho + blend (gitignored)
├── scripts/                    # pipeline numerado 00→06 (ver comandos)
├── src/
│   ├── frontend/               # SPA inyectado: inject.py + templates/
│   │   └── templates/          #   effects.html (Lenis/GSAP/Three.js),
│   │                           #   bracket.html (cuadro interactivo + SVG)
│   └── mundial/
│       ├── auth.py             # RBAC viewer/admin (modal, fail-closed)
│       ├── features/           # build.py (features batch), elo.py
│       ├── models/             # baseline.py (FEATURES, clf), poisson.py
│       ├── predict/            # engine (PredictionEngine), montecarlo.py
│       └── live/               # store.py / state.py / online.py / engine.py
├── tests/                      # pytest (27 verdes) — usan tmp_path, no data/
├── docs/ARCHITECTURE_SPEC.md   # FUENTE DE VERDAD (SDD)
├── logs/app.log                # logging runtime (gitignored)
└── .github/workflows/ci.yml    # ruff + bandit + pip-audit + pytest
```

## 3. Comandos frecuentes

```bash
# UI (lo único necesario durante el Mundial)
.venv/Scripts/python.exe -m streamlit run app.py

# Calidad (lo que corre el CI)
.venv/Scripts/ruff check app.py src/ scripts/ tests/
.venv/Scripts/bandit -r app.py src/ -q
.venv/Scripts/pip-audit --skip-editable
.venv/Scripts/python -m pytest          # pythonpath=src vía pyproject

# Refrescar data lake + reentrenar (martj42 se actualiza a diario)
python scripts/00_download_tier0.py     # Bronze
python scripts/01_build_silver.py       # Silver
python scripts/02_build_features.py     # Gold (anti-leakage checks)
python scripts/04_train_final.py        # artifacts.joblib
python scripts/05_consolidate_live.py   # silver+live consolidado
python scripts/06_build_rosters.py      # rosters Gold

# Smoke test sin browser (ejecuta TODO el script en bare mode)
.venv/Scripts/python -c "import app"
```

## 4. Convenciones de código

- Python 3.12, type hints en firmas públicas (`from __future__ import
  annotations`), docstrings en español explicando el PORQUÉ.
- Ruff limpio (config en `pyproject.toml`; `app.py` exenta de E402 por el
  `sys.path.insert` necesario). Líneas ≤ 100.
- SOLID pragmático: módulos chicos con una responsabilidad (store =
  persistencia, state = ajustes Elo, online = corrección bayesiana,
  engine = orquestación). La UI no contiene lógica de modelo.
- Sorts con `kind="stable"` (determinismo). Funciones puras donde se pueda
  (`rank_group`, `sanitize_text`) para testear sin Streamlit.
- Streamlit: cachés siempre keyed por `STORE.token()` cuando dependen de
  data live (invariante U2). Widgets con `key` explícito en formularios.

## 5. Motor ML (resumen para no romperlo)

- **Features pre-partido** (Gold): Elo (K=30 plano, base 1500), forma
  (pts/gf/gc últimos 5), H2H, descanso, experiencia. `PredictionEngine`
  replica EXACTAMENTE las features batch de `features/build.py` —
  verificado numéricamente; si cambias una feature, cambia ambos lados.
- **Ensemble**: `blend(0.8)·predict_proba(clf) + 0.2·outcome_probs(
  score_matrix(λh, λa, ρ=-0.15))`, clases en orden `['A','D','H']`.
  La matriz de marcadores se reescala para que sus marginales 1X2
  coincidan con el ensemble (coherencia card/porcentajes).
- **Capa live** (sin reentrenar): momentum K_LIVE=45 sobre el Elo,
  suspensiones FIFA (roja o 2 amarillas; amarillas se limpian tras
  cuartos), lesiones (-9/-7/-14 Elo, tope -45), corrección bayesiana con
  shrinkage (γ goles prior 20 xG, empates prior 25, altitud ≥1400m
  prior 1.05; xG blend 0.65 goles/0.35 xG). NUNCA `model.fit()` en vivo.
- **Qué usa el modelo de la ingesta**: goles/xG → momentum y γ; tarjetas →
  suspensiones; lesiones → ajuste Elo. **Clima y formación = SOLO
  metadatos** (así está etiquetado en la UI; si algún día entran como
  features → reentrenar y actualizar spec §4).
- **Bracket determinista (U4)**: entrantes a R32 = ocupante modal del
  Monte Carlo; desde ahí avanza el de `p_advances > 50%` en CADA cruce
  (`engine.predict_match`), o el ganador real ingresado. Las marginales de
  `slot_stats` NO componen entre rondas; los pct visibles de cada llave
  son P(avanzar en ese cruce) (U4-display), la ocupación marginal vive en
  "Alt:" del modo foco.
- **Monte Carlo**: `rank_group` aplica desempate FIFA Pts→DG→GF→H2H→azar;
  mejores 8 terceros greedy por slots permitidos; prórroga/penales por Elo.
- **Backtesting (tab Auditoría, spec §9)**: SIEMPRE desde features Gold +
  artifacts (pre-partido); nunca con el Elo actual del engine (leakage).

## 6. Seguridad

- RBAC fail-closed: sin secrets ⇒ viewer permanente. Clave en
  `st.secrets["admin_password"]` (top-level; `[auth]` aceptado por compat).
  `hmac.compare_digest`. El tab de ingesta NO se construye para viewers.
- Invariante R2: el login nunca muestra detalles de configuración ni
  distingue clave-incorrecta vs auth-sin-configurar ("Acceso denegado.");
  el motivo va solo a `logs/app.log`.
- Invariante S3: `LiveStore.add_match` sanitiza texto libre
  (`sanitize_text`: control chars, prefijos de fórmula `=+-@`, 120 chars)
  — anti CSV-injection. El display además escapa HTML (`esc()`).
- `.streamlit/secrets.toml` y `logs/` gitignored; verificado que nunca se
  versionaron. La clave NUNCA va en archivos versionados ni en la UI.
- CI (`.github/workflows/ci.yml`): ruff + bandit (0 hallazgos) +
  pip-audit (0 CVEs al 2026-06-10) + pytest.

## 7. Despliegue (Streamlit Community Cloud)

URL pública: https://oraculo-2026.streamlit.app/

### Secrets requeridos (share.streamlit.io → app → Settings → Secrets)
```toml
admin_password = "clave-fuerte-aqui"

# PAT GitHub para persistencia de data/live/ (Opción A — ver github_sync.py)
github_token = "ghp_..."
# github_repo = "NicoBJ1906/mundial-2026-ml"   # por defecto
# github_branch = "main"                          # por defecto
```

### Cómo funciona la persistencia (Opción A — GitHub como store)
- Cuando el admin ingresa un resultado, `LiveStore.add_match()` escribe los
  4 CSVs localmente y luego llama a `github_sync.sync_live_files()`.
- Sync usa la **Git Trees API**: un único commit atómico para los 4 archivos
  → sin commits intermedios, sin race conditions.
- Streamlit Cloud detecta el push y redespliega (~60 s).
- Si GitHub no está disponible, se loguea WARNING y la app sigue sin caerse.
  Los datos quedan en el filesystem del servidor Cloud (efímero) hasta el
  próximo reinicio, y el próximo partido intentará sincronizar de nuevo.
- Token: Fine-grained PAT, permiso `Contents: Read and write` solo en este repo.

### Archivos versionados para Cloud
Los 7 archivos de `data/` y `models/` están en el repo con excepción explícita
en `.gitignore` (ver sección). `data/live/*.csv` se sincronizan vía API GitHub,
NO con `git add`.

- `st.components.v1.html` está deprecado (fecha vencida 2026-06-01) — migrar
  a `st.iframe` en próxima sesión.

## 8. Decisiones clave / gotchas (no romper)

- Banderas vía flagcdn (`FLAG_ISO`); solo anchos w40/80/160/320. En cards
  se normalizan a 40×26 `object-fit: cover` (alturas nativas distintas).
- Emojis de bandera NO renderizan en Windows; Material icons necesitan
  excluirse del font-override global (`span[data-testid="stIconMaterial"]`
  → 'Material Symbols Rounded', si no se ve "keyboard_arrow_down" texto).
- Theming: vars propias (`--bg/--text/--accent/…`) inyectadas en `:root`
  por `inject_theme()`; NO usar las nativas de Streamlit (`--text-color`)
  — no cambian con el toggle runtime. Los iframes parsean el último
  `<style>` con `--bg:` del BODY padre + MutationObserver (characterData);
  además copian el `color-scheme` del padre (si difiere, el browser fuerza
  canvas opaco = recuadro negro).
- Modales: baseweb los porta con tema de config.toml (dark) → CSS fuerza
  `var(--surface-solid)` en `stDialog`/`[data-baseweb="modal"]`/KaTeX.
- Paneles que envuelven widgets: `st.container(border=True)` (los `<div>`
  abiertos/cerrados en markdowns separados NO contienen nada).
- Tabla de resultados ingresados: nombres de columna ÚNICOS (con
  duplicados `r[c]` devuelve una Series y la celda imprime basura).
- Si tras editar módulos de `src/` la app "pierde" funciones nuevas:
  reiniciar streamlit (el proceso viejo cachea los imports).
- Responsive móvil (invariante U5 del spec): TODO el CSS móvil vive en un
  único `@media (max-width: 768px)` al final de `BASE_CSS` — nunca tocar
  reglas desktop para arreglar móvil. Culpables históricos del overflow-x:
  tab-list pill con `width: fit-content` (7 tabs > viewport), `min-width`
  fijos de `.audit-row`, y el `#tree` del bracket (se arregla con
  `min-width` interno + scroll-x DENTRO del iframe, jamás en la página).
- Empates: el modelo les da probabilidad correcta pero casi nunca argmax —
  esperado, no es bug. Expectativa honesta: 55-60% acierto 1X2 es el techo
  del estado del arte; el valor real es la calibración.
- `gh` CLI NO instalado: API REST con `$GITHUB_TOKEN`; push con
  `git -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"`
  (nunca persistir el token en .git/config). Repo: NicoBJ1906/mundial-2026-ml.

## 9. Estado y métricas (modelo del 2026-06-10, datos al 2026-06-09)

- Entrenado con 15.618 partidos · test temporal 2022+ (4.541): acierto
  1X2 **0.602**, log-loss 0.867 · hold-out Mundial 2022: 0.547 ·
  calibración por buckets casi perfecta · marcador exacto card 13.1%.
- 27 tests pytest verdes · ruff/bandit/pip-audit limpios.
- Tabs: Próximos (jornadas reales) · [Ingresar resultado] · Líderes ·
  Cuadro (determinista + conectores SVG) · Eliminatorias · Camino al
  título · Tablas · Auditoría (backtesting últimos 5 por selección).

## 10. Historial condensado de sesiones

- **06-09**: capa live completa (store/state/online/engine), temas
  dark/light, spec SDD, 19 tests, XAI, H2H FIFA, publicación GitHub.
- **06-10 (mañana)**: frontend SPA (Lenis/GSAP/Three.js), bracket
  interactivo, RBAC, rosters Gold, XAI pedagógico (st.latex), 23 tests.
- **06-10 (tarde)**: login a modal (sidebar roto por header oculto),
  bracket determinista U4 + conectores SVG, jornadas reales, fix
  color-scheme iframes, logging, pipeline corrido (modelo 15.618),
  secrets formato Cloud, CI + ruff/bandit/pip-audit, fix CSV-injection
  (S3), login sin fugas (R2), badges "solo metadatos", pct del bracket =
  P(avanza del cruce) (U4-display), tab Auditoría (spec §9).

## 11. Pendiente / ideas

- Decidir estrategia de artifacts/data para Community Cloud (ver §7).
- Migrar `components.html` → `st.iframe`.
- Sprints 3-4 del PLAN (API-Football, StatsBomb xG) sin empezar.
- `evaluate/` vacío (los chequeos viven en scripts y en el tab Auditoría).
