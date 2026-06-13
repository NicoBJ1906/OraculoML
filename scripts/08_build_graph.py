"""Grafo de conocimiento visual del proyecto (estilo Graphify/Neo4j).

Analiza el código con AST (imports internos + referencias a datos/modelos)
y genera docs/project_graph.html: red interactiva force-directed
(vis-network) con nodos coloreados por tipo y aristas semánticas.

Uso:
    python scripts/08_build_graph.py
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

COLORS = {"ui": "#ff2d75", "modulo": "#19c8ff", "script": "#2bff88",
          "dataset": "#ffd31a", "modelo": "#b13bff", "frontend": "#ff8b2d"}

# patrones de archivos de datos/modelos referenciados en código
DATA_PAT = re.compile(
    r'["\']([\w./\\-]+\.(?:parquet|joblib|csv|json|html))["\']')

# rutas de datos que NO son nodos interesantes (genéricos/temporales)
SKIP_DATA = ("secrets", "log", ".gitkeep")


def module_name(p: Path) -> str:
    rel = p.relative_to(SRC) if SRC in p.parents else p.relative_to(ROOT)
    return str(rel.with_suffix("")).replace("\\", "/")


def scan(py: Path) -> tuple[set[str], set[str]]:
    """(imports internos 'mundial.x.y'/'frontend', datos referenciados)."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    imps: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and \
                n.module.split(".")[0] in ("mundial", "frontend"):
            imps.add(n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in ("mundial", "frontend"):
                    imps.add(a.name)
    text = py.read_text(encoding="utf-8")
    data = {m.group(1).replace("\\\\", "/").split("/")[-1]
            for m in DATA_PAT.finditer(text)
            if not any(s in m.group(1).lower() for s in SKIP_DATA)}
    return imps, data


def main() -> None:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add(nid: str, kind: str, size: int = 18) -> None:
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": nid.split("/")[-1],
                          "title": nid, "color": COLORS[kind],
                          "shape": "dot", "size": size,
                          "group": kind}

    files = ([ROOT / "app.py"] + sorted(SRC.rglob("*.py"))
             + sorted((ROOT / "scripts").glob("*.py")))
    for py in files:
        if "__init__" in py.name or "__pycache__" in str(py):
            continue
        if py.name == "app.py":
            me, kind = "app.py", "ui"
        elif "scripts" in str(py.parent):
            me, kind = f"scripts/{py.stem}", "script"
        else:
            me, kind = module_name(py), "modulo"
        add(me, kind, 30 if kind == "ui" else 18)
        imps, data = scan(py)
        for imp in imps:
            tgt = imp.replace("mundial.", "mundial/").replace(".", "/")
            add(tgt, "modulo")
            edges.append({"from": me, "to": tgt, "arrows": "to",
                          "title": "importa"})
        for d in data:
            k = ("modelo" if d.endswith(".joblib")
                 else "frontend" if d.endswith(".html") else "dataset")
            add(d, k, 14)
            edges.append({"from": me, "to": d, "arrows": "to", "dashes": True,
                          "title": "lee/escribe", "color": {"opacity": .45}})

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>OráculoML — Red neuronal del proyecto</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<style>body{{margin:0;background:#03040c;font-family:Segoe UI,sans-serif;
overflow:hidden}}
#stars,#net{{position:fixed;inset:0;height:100vh;width:100vw}}
#hud{{position:fixed;top:14px;left:16px;color:#e6edf3;z-index:9;
text-shadow:0 0 12px rgba(25,200,255,.8)}}#hud h2{{margin:0 0 6px}}
.lg{{display:inline-block;margin-right:14px;font-size:.85rem}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:99px;
margin-right:5px;box-shadow:0 0 8px currentColor}}</style></head><body>
<canvas id="stars"></canvas><div id="net"></div>
<div id="hud"><h2>🧠 OráculoML — red neuronal del proyecto</h2>
<span class="lg"><i style="background:#ff2d75;color:#ff2d75"></i>UI</span>
<span class="lg"><i style="background:#19c8ff;color:#19c8ff"></i>módulo</span>
<span class="lg"><i style="background:#2bff88;color:#2bff88"></i>pipeline</span>
<span class="lg"><i style="background:#ffd31a;color:#ffd31a"></i>dataset</span>
<span class="lg"><i style="background:#b13bff;color:#b13bff"></i>modelo</span>
<span class="lg"><i style="background:#ff8b2d;color:#ff8b2d"></i>frontend</span>
</div><script>
/* ---- campo de estrellas con parallax ---- */
const sc = document.getElementById('stars'), sx = sc.getContext('2d');
sc.width = innerWidth; sc.height = innerHeight;
const stars = Array.from({{length: 240}}, () => ({{
  x: Math.random() * sc.width, y: Math.random() * sc.height,
  r: Math.random() * 1.4 + .2, tw: Math.random() * 6.28,
  v: Math.random() * .12 + .02}}));
function drawStars(t) {{
  sx.clearRect(0, 0, sc.width, sc.height);
  for (const s of stars) {{
    s.x = (s.x + s.v) % sc.width;
    const a = .35 + .65 * Math.abs(Math.sin(t / 900 + s.tw));
    sx.beginPath(); sx.arc(s.x, s.y, s.r, 0, 6.28);
    sx.fillStyle = `rgba(180,210,255,${{a}})`; sx.fill();
  }}
}}
/* ---- red ---- */
const nodes = {json.dumps(list(nodes.values()))};
const edges = {json.dumps(edges)};
nodes.forEach(n => {{ n.borderWidth = 0;
  n.shadow = {{enabled: true, color: n.color, size: 26, x: 0, y: 0}};
  n.color = {{background: n.color,
              highlight: {{background: '#ffffff', border: n.color}}}}; }});
const net = new vis.Network(document.getElementById('net'),
  {{nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)}},
  {{physics: {{barnesHut: {{gravitationalConstant: -10500,
      springLength: 135, damping: .12}},
    stabilization: {{iterations: 200}}}},
    nodes: {{font: {{color: '#cfe6ff', size: 12,
                     strokeWidth: 4, strokeColor: '#03040c'}}}},
    edges: {{color: {{color: 'rgba(80,140,210,.28)'}}, width: 1,
             smooth: {{type: 'continuous'}}}},
    interaction: {{hover: true}}}});
/* ---- pulsos sinápticos viajando por las aristas ---- */
const pulses = edges.map((e, i) => ({{e, off: (i * .37) % 1,
  speed: .0018 + Math.random() * .0028,
  hue: ['#19c8ff', '#ff2d75', '#2bff88', '#ffd31a'][i % 4]}}));
net.on('afterDrawing', ctx => {{
  const t = performance.now();
  for (const p of pulses) {{
    const a = net.getPosition(p.e.from), b = net.getPosition(p.e.to);
    const k = (t * p.speed + p.off) % 1;
    const x = a.x + (b.x - a.x) * k, y = a.y + (b.y - a.y) * k;
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 6.28);
    ctx.fillStyle = p.hue;
    ctx.shadowColor = p.hue; ctx.shadowBlur = 14;
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.beginPath();                      // estela
    ctx.arc(a.x + (b.x - a.x) * Math.max(k - .05, 0),
            a.y + (b.y - a.y) * Math.max(k - .05, 0), 1.2, 0, 6.28);
    ctx.fillStyle = p.hue + '66'; ctx.fill();
  }}
}});
/* ---- latido: el hub respira ---- */
(function loop(t) {{ drawStars(t || 0); net.redraw();
  requestAnimationFrame(loop); }})();
</script></body></html>"""
    out = ROOT / "docs" / "project_graph.html"
    out.write_text(html, encoding="utf-8")
    print(f"nodos: {len(nodes)}  aristas: {len(edges)}  -> {out}")


if __name__ == "__main__":
    main()
