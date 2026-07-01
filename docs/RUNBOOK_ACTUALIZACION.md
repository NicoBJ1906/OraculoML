# Runbook — Actualizar el Oráculo con partidos jugados

Paso a paso para mantener la app al día durante el Mundial. Todo se hace sobre
los CSV de `data/live/` y se publica al repo con la **Git Trees API** (NO
`git push`, ver §5).

Rutas: proyecto en `D:\Proyects\mundial-2026-ml`; Python = `.venv/Scripts/python.exe`.

---

## 0. Qué mueve qué (leer antes de actualizar)

| CSV en `data/live/` | Alimenta | ¿Afecta la PREDICCIÓN? |
|---|---|---|
| `live_results.csv` | Elo / forma / momentum (online) | **Sí — es lo único que la mueve** |
| `live_discipline.csv` | suspensiones (−9 Elo, se consumen al partido siguiente) | Sí, marginal |
| `live_injuries.csv` | bajas por lesión | Sí, marginal |
| `live_players.csv` | tab **Líderes** (goleadores/asistencias) | **No** — solo cosmético |

> Cargar partidos **NO reentrena** el ensemble; solo actualiza el *estado*
> (online learning). El modelo da **probabilidades, no certezas**. Para reducir
> el error sistemático hay que **recalibrar pesos + backtest**, no cargar más datos.

---

## 1. Averiguar qué partidos faltan

```bash
.venv/Scripts/python.exe - <<'PY'
import csv, json, unicodedata
def slug(s):
    s=unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode()
    return s.lower().replace(" ","-")
have=set(r["match_id"] for r in csv.DictReader(open("data/live/live_results.csv",encoding="utf-8")))
wc=json.load(open("data/raw/worldcup2026/worldcup.json",encoding="utf-8"))
for m in wc["matches"]:
    d=m["date"]; t1=m.get("team1"); t2=m.get("team2")
    print(d, m["round"], t1, "vs", t2)   # comparar contra 'have'
PY
```

Los slots de eliminatorias vienen como placeholders (`2A vs 2B`, `1E vs 3A/B/C/D/F`);
los cruces reales se resuelven solos al completar los grupos (ver §4).

## 2. Buscar resultados REALES (nunca inventar)

Fuentes fiables: FIFA Match Centre, ESPN, Yahoo Sports (bracket), Wikipedia.
Buscar marcador final + penales si hubo. Para eliminatorias, Yahoo
"Round of 32 full bracket" da los 16 cruces con resultado.

## 3. Cargar en los CSV (formato exacto)

`match_id = {AAAAMMDD}_{slug(home)}_{slug(away)}` (slug = minúsculas, espacios→`-`,
sin acentos; ej. `DR Congo`→`dr-congo`, `Cape Verde`→`cape-verde`).

**Resultados** — `live_results.csv`
`match_id,date,home_team,away_team,home_score,away_score,neutral,stage,ko_winner,xg_home,xg_away,weather,formation_home,formation_away`
- `neutral=True` salvo que juegue anfitrión (USA/Mexico/Canada) en casa.
- `stage`: `group` o `R32`/`R16`/`QF`/`SF`/`F`.
- En KO con empate a 90' + penales: poner el marcador de 90' y **`ko_winner`** con el que pasó.

**Goleadores** — `live_players.csv`
`match_id,date,team,player,event,minute` · `event` ∈ `goal|penalty|own_goal|assist`.
- Usar **exactamente** el mismo nombre que ya exista (ej. `Kylian Mbappé`) para que sume.
- El minuto no afecta el ranking (la tab cuenta eventos por jugador); si no se
  conoce, aproximar y seguir.

**Tarjetas** — `live_discipline.csv`: `match_id,date,team,player,card,minute` (`card`=`red|yellow`).
Las amarillas se limpian tras la fase de grupos y tras cuartos; a KO solo
arrastran las rojas, que suspenden el **partido siguiente**.

**Lesiones** — `live_injuries.csv`: `match_id,date,team,player,severity`.

> Escribir siempre con `encoding="utf-8"` (la consola Windows es cp1252 y
> revienta con acentos en el `print`, no en el archivo).

## 4. Consolidar y verificar

```bash
.venv/Scripts/python.exe scripts/05_consolidate_live.py   # silver + live
.venv/Scripts/python.exe scripts/06_build_rosters.py      # si cambian rosters
.venv/Scripts/ruff check src/ app.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -c "import app"                  # smoke (corre bare mode)
```

Verificar cruces KO resueltos (deben coincidir con el bracket oficial):
```bash
.venv/Scripts/python.exe - <<'PY'
import sys; sys.path.insert(0,"src")
import app
bp=app.build_bracket_payload(app.STORE.token(), 2000)
for r in bp["rounds"]:
    for m in r["matches"]:
        if m["t1"] and m["t2"]:
            print(r["label"], m["t1"]["team"],"vs",m["t2"]["team"],"->",m["win"], "JUGADO" if m["played"] else "proy.")
PY
```

**Asignación de terceros (bug histórico):** `_assign_thirds` en
`src/mundial/predict/montecarlo.py` usaba un greedy que NO replica la tabla FIFA
→ cruces R32 falsos (p.ej. Germany-Sweden). La combinación real 2026 (terceros
eliminados A,C,G,H) está en `THIRD_ALLOCATION`. Si cambia la combinación, agregar
la fila correspondiente (clave = frozenset de los 4 grupos sin tercero; valor =
{frozenset(allowed del slot): letra_del_grupo_tercero}).

## 5. Publicar al repo (Git Trees API, NO git push)

`git push` falla con **403** (cuenta autenticada `NicolasBJ19` ≠ dueño
`NicoBJ1906`). Usar el `GITHUB_TOKEN` del entorno vía la Trees API reutilizando
`_request` de `github_sync.py`:

```bash
.venv/Scripts/python.exe - <<'PY'
import os,sys,base64; sys.path.insert(0,"src")
from pathlib import Path
from mundial.live.github_sync import _request,_API_BASE
tok=os.environ["GITHUB_TOKEN"]; repo="NicoBJ1906/OraculoML"; branch="main"
base=f"{_API_BASE}/repos/{repo}"
files=["data/live/live_results.csv","data/live/live_players.csv"]  # los que cambiaron
head=_request("GET",f"{base}/git/refs/heads/{branch}",tok)["object"]["sha"]
bt=_request("GET",f"{base}/git/commits/{head}",tok)["tree"]["sha"]
items=[]
for f in files:
    b=_request("POST",f"{base}/git/blobs",tok,{"content":base64.b64encode(Path(f).read_bytes()).decode(),"encoding":"base64"})
    items.append({"path":f,"mode":"100644","type":"blob","sha":b["sha"]})
tr=_request("POST",f"{base}/git/trees",tok,{"base_tree":bt,"tree":items})
c=_request("POST",f"{base}/git/commits",tok,{"message":"live: <describir>","tree":tr["sha"],"parents":[head]})
_request("PATCH",f"{base}/git/refs/heads/{branch}",tok,{"sha":c["sha"]})
print("OK commit",c["sha"][:7])
PY
```

Luego alinear el local con el remoto (evita divergencia):
```bash
git fetch origin -q && git reset --hard origin/main
```

Streamlit Cloud (`oraculoml-2026.streamlit.app`) redespliega solo en ~60 s.
Si mezcla módulos viejos tras el push: **Reboot app** en el panel de Cloud.

## 6. Medir aciertos (opcional)

Comparar favorito pre-partido del modelo vs. ganador real de los KO ya jugados
(ver `build_bracket_payload` + `engine.predict_match`). Referencia histórica:
techo realista ~53-60% acc; en R32 2026 el modelo acertó 6/7 (falló Germany,
cayó en penales — varianza irreducible).
