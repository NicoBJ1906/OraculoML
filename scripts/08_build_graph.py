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

COLORS = {"ui": "#e63946", "modulo": "#457b9d", "script": "#2a9d8f",
          "dataset": "#e9c46a", "modelo": "#9b5de5", "frontend": "#f4a261"}

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
<title>OráculoML — Grafo del proyecto</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<style>body{{margin:0;background:#0d1117;font-family:Segoe UI,sans-serif}}
#net{{height:100vh}}#hud{{position:fixed;top:14px;left:16px;color:#e6edf3;
z-index:9}}#hud h2{{margin:0 0 6px}}.lg{{display:inline-block;margin-right:14px;
font-size:.85rem}}.lg i{{display:inline-block;width:10px;height:10px;
border-radius:99px;margin-right:5px}}</style></head><body>
<div id="hud"><h2>⚽ OráculoML — red de conocimiento</h2>
<span class="lg"><i style="background:#e63946"></i>UI</span>
<span class="lg"><i style="background:#457b9d"></i>módulo</span>
<span class="lg"><i style="background:#2a9d8f"></i>script pipeline</span>
<span class="lg"><i style="background:#e9c46a"></i>dataset</span>
<span class="lg"><i style="background:#9b5de5"></i>modelo</span>
<span class="lg"><i style="background:#f4a261"></i>frontend</span></div>
<div id="net"></div><script>
const nodes={json.dumps(list(nodes.values()))};
const edges={json.dumps(edges)};
new vis.Network(document.getElementById('net'),
  {{nodes:new vis.DataSet(nodes), edges:new vis.DataSet(edges)}},
  {{physics:{{barnesHut:{{gravitationalConstant:-9000,springLength:130}},
    stabilization:{{iterations:220}}}},
    nodes:{{font:{{color:'#e6edf3',size:13}},borderWidth:0}},
    edges:{{color:{{color:'#6e7681',opacity:.7}},smooth:true}},
    interaction:{{hover:true}}}});
</script></body></html>"""
    out = ROOT / "docs" / "project_graph.html"
    out.write_text(html, encoding="utf-8")
    print(f"nodos: {len(nodes)}  aristas: {len(edges)}  -> {out}")


if __name__ == "__main__":
    main()
