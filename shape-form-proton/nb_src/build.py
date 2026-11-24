"""A végső notebook összeállítása és lefuttatása: `nb_src/cells_*.py` fájlokból (# %% [markdown] / # %% cellák) -> .ipynb kimenetekkel.

Használat:  python nb_src/build.py <kimenet.ipynb> cells_a.py [cells_b.py ...] [--no-run]
"""
import re, sys, os
import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse(path):
    cells, kind, buf = [], None, []
    for line in open(path, encoding="utf-8").read().split("\n"):
        m = re.match(r"^# %%\s*(\[markdown\])?\s*$", line)
        if m:
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("markdown" if m.group(1) else "code"), []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    out = []
    for kind, src in cells:
        if kind == "markdown":                       # a markdown cellák soronként "# "-rel kezdődnek
            src = "\n".join(l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l) for l in src.split("\n"))
            out.append(nbformat.v4.new_markdown_cell(src))
        else:
            out.append(nbformat.v4.new_code_cell(src))
    return out


args = [a for a in sys.argv[1:] if not a.startswith("--")]
out, parts = args[0], args[1:]
nb = nbformat.v4.new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
for p in parts:
    nb.cells += parse(os.path.join(HERE, p))
if "--no-run" not in sys.argv:
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
nbformat.write(nb, os.path.join(ROOT, out))
print("written", out, "cells", len(nb.cells))
