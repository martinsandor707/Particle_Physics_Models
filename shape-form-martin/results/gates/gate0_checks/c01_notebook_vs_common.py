"""Gate 0 helper: is the metric code that produced the reference numbers (code cells of
shape-form-proton/00_common.ipynb, exec'd by protonlab.load_common) the same as shape-form-martin/common.py?
Compares the source of every top-level function/class and the grid constants, textually (AST-normalised).
Read-only; nothing is executed from the notebook."""
import ast
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
nb = json.loads((REPO / "shape-form-proton" / "00_common.ipynb").read_text())
nb_src = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# notebook magics / top-level display code are irrelevant for function definitions; strip IPython magics
nb_src = "\n".join(l for l in nb_src.splitlines() if not l.lstrip().startswith(("%", "!")))
py_src = (REPO / "shape-form-martin" / "common.py").read_text()


def defs(src):
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, (ast.FunctionDef,)) and ast.get_docstring(node) is not None:
                pass
            out[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out["=" + n.id] = ast.dump(node.value, include_attributes=False)
    return out


a, b = defs(nb_src), defs(py_src)
metric_names = ["fold", "event_nll", "test_nll", "locate", "rotation", "_ranges", "Showers", "load_data",
                "detector_geometry", "FieldModel", "Formula", "select_architecture", "select_degree",
                "=Z_EDGES", "=R_EDGES", "=DZ", "=ZC", "=RC", "=LOG_VOL", "=ZZ", "=RR",
                "=Z_MIN", "=Z_MAX", "=N_Z", "=R_MAX", "=N_R", "=N_FOLD", "=N_FOLD_EVAL", "=TIME_CUT_NS", "=SEED", "=DEVICE"]
print(f"definitions: notebook {len(a)}, common.py {len(b)}")
print("only in notebook:", sorted(set(a) - set(b)))
print("only in common.py:", sorted(set(b) - set(a)))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print("present in both but different AST:", diff)
print()
for k in metric_names:
    st = "MISSING" if (k not in a or k not in b) else ("identical" if a[k] == b[k] else "DIFFERENT")
    print(f"  {k:22s} {st}")

# git: common.py at HEAD (old path) vs working copy
old = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:shape-form-original/common.py"], capture_output=True, text=True).stdout
print("\ncommon.py working copy == HEAD:shape-form-original/common.py :", old == py_src)
