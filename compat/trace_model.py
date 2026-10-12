"""Run one text-to-cad model on keel.build123d and log fillet/chamfer/extrude calls and failures."""
import importlib, os, sys, tempfile, traceback
sys.path.insert(0, os.path.dirname(__file__))
import t2c_harness as h
from pathlib import Path
standin = Path(tempfile.mkdtemp()); h._write_standin(standin)
os.environ["T2C_BACKEND"] = sys.argv[3] if len(sys.argv) > 3 else "keel"
sys.path[:0] = [str(standin), sys.argv[1], str(Path(__file__).resolve().parents[1] / "src")]
import importlib as _il
bd = _il.import_module("keel.build123d" if os.environ["T2C_BACKEND"] == "keel" else "build123d")
for name in ("fillet", "chamfer", "extrude"):
    orig = getattr(bd, name)
    def wrapped(*a, _orig=orig, _name=name, **k):
        try:
            r = _orig(*a, **k)
            bp = None
            try:
                ctx = bd.BuildPart._get_context() if hasattr(bd.BuildPart, "_get_context") else (bd._builders[-1] if bd._builders else None)
                bp = ctx.part.volume if ctx is not None and hasattr(ctx, "part") and ctx.part is not None else None
            except Exception:
                pass
            print(f"  {_name}({len(a[0]) if a and isinstance(a[0], list) else ''}) ok -> vol {r.volume:.2f}  builder part {bp if bp is None else round(bp, 2)}")
            return r
        except Exception as e:
            print(f"  {_name} FAILED: {type(e).__name__}: {str(e)[:160]}")
            raise
    setattr(bd, name, wrapped)
import cadgen
mod = importlib.import_module(sys.argv[2])
fn = cadgen._REGISTRY[0][2]
print("result volume", fn().volume)
