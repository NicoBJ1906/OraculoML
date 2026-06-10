# Mejoras UI — DeepSeek v4 (09 Jun 2026)

## 1. CSS Global completo

**Antes**: Inter + Space Grotesk, glassmorphism básico, transiciones simples.

**Después**: Poppins (Google Fonts) en toda la app, scrollbar personalizada,
glassmorphism mejorado (`blur(28px) saturate(160%)`), transiciones
`cubic-bezier(.16,1,.3,1)` premium en hover/active, focus rings en inputs,
slider con glow y escala al hover.

### Archivos tocados
- `app.py` — reemplazo completo de `BASE_CSS` (lineas 98-406)

## 2. match_card — diseño tipográfico

**Antes**: Barra horizontal de colores (rojo/gris/azul) + score predicho "2-1".

**Después**: Diseño puramente tipográfico. Muestra solo:
- `Local XX% · Empate XX% · Visit. XX%` con fuente bold
- Flags de equipos
- Botón "📊 Explicar pronóstico" por card

### Archivos tocados
- `app.py` — función `match_card` (lineas 431-459)

## 3. Modal XAI con st.dialog

**Antes**: Expander con tabla de ajustes Elo.

**Después**: Modal nativo `st.dialog` con glassmorphism que desglosa:
- Elo base, momentum, sanciones/lesiones, ajuste total, Elo efectivo
- λ Poisson (goles esperados) y top-5 marcadores
- Correcciones globales: γ (ritmo goles), draw_mult, alt_mult
- Por equipo: detalle de cada sanción/lesión con etiqueta

### Archivos tocados
- `app.py` — función `xai_dialog` (lineas 642-794)

## 4. Tab Ingresar Resultados — rediseño

**Antes**: Una sola columna con `st.divider`, inputs sueltos.

**Después**: Dos paneles lado a lado (`st.columns`) con tarjetas `form-card`:
- **Fase de grupos**: selector de fixture, marcador, contexto
- **Eliminatoria/manual**: selectores libres con fecha
- Tooltips en inputs de goles y xG
- Tabla de resultados ingresados abajo

### Archivos tocados
- `app.py` — reemplazo completo del bloque `tab_result` (lineas 823-901)

## 5. Tab Líderes — tarjetas animadas

**Antes**: Tablas HTML estáticas (`tbl()`).

**Después**: Tarjetas `leader-card` con hover animation (`translateY(-4px) scale(1.02)`),
top-10 con medallas numeradas. Gradiente sutíl de acento en el fondo.

### Archivos tocados
- `app.py` — reemplazo del bloque `tab_leaders` (lineas 909-1007)
- CSS añadido: `.leader-card`, `.leader-num`, `.leader-name`, `.leader-stat`, `.leader-badge`

## 6. Bracket — espaciado correcto del árbol

**Antes**: `justify-content: space-around` que distribuía desigualmente las
llaves (R32 apretado abajo, Final suelto arriba).

**Después**: Cálculo algebraico de márgenes:
- `MATCH_H = 72px`, `MAX_M = 16`, `H_TOTAL = 1152px`
- Para cada partido en cada ronda: `margin-top` y `margin-bottom` calculados
  para que los centros estén en las posiciones correctas del árbol
  `(i + 0.5) / n_m * H_TOTAL`

### Archivos tocados
- `app.py` — bloque `tab_bracket` (lineas 1008-1070)
- CSS: eliminado `min-height`, `justify-content`, `gap` en `.b-col`

## 7. Podium (Camino al título) — colores restaurados

**Antes**: Las probabilidades de final/semis perdieron el color rojo/azul porque
se eliminaron las reglas `.mc-probs span:first-child/last-child`.

**Después**: Restauradas las reglas CSS de color del `mc-probs` legacy.

### Archivos tocados
- `app.py` — CSS añadido `.mc-probs { ... }` con `span:first-child {color: var(--accent)}`
  y `span:last-child {color: var(--bar-a1)}`

## 8. Slider (Días a mostrar) — rediseñado

**Antes**: Slider plano sin estilo, thumb pequeño.

**Después**: Thumb 20×20 con glow y borde, track degradado rojo-anaranjado,
hover scale 1.15, fuente semibold en la label.

### Archivos tocados
- `app.py` — CSS del slider reemplazado

## 9. Expander — padding interno

**Antes**: El contenido del expander estaba pegado al borde.

**Después**: Padding añadido a `stExpanderDetails` y summary.

## 10. SPEC actualizada

`docs/ARCHITECTURE_SPEC.md` v1.0 → v1.1
- Sección 5 reescrita con todos los nuevos componentes UI
- Invariantes U3 (XAI modal)
- Detalle de match_card, xai_dialog, leader-card, form-card, bracket

## Lo que NO se cambió (intacto)
- Todo el motor ML: PredictionEngine, LiveEngine, TournamentSimulator,
  OnlineCorrector, TournamentState, Elo, Poisson, Logistic
- Toda la capa de datos: LiveStore, transform, ingest, features, build
- Tests existentes (19 siguen pasando)
- `config/settings.yaml`, rutas, .streamlit/config.toml
