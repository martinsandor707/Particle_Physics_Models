"""pytest setup: make ``common`` (shape-form-martin/) and ``sfm`` (shape-form-martin/src) importable
and never write bytecode (imports from shape-form-proton/ must not create .pyc files there)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_MARTIN = Path(__file__).resolve().parents[1]
for _p in (_MARTIN / "src", _MARTIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
