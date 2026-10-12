"""Run text-to-cad model scripts, unmodified, against a chosen build123d implementation.

    python t2c_harness.py --backend build123d --models <dir> --out ref.json   # real build123d
    python t2c_harness.py --backend keel      --models <dir> --out keel.json  # keel.build123d

A stand-in ``cadgen`` package makes ``from cadgen import build123d as bd`` resolve to the
chosen backend. Each ``@step``/``@glb``/... function runs in its own subprocess (so one
crash or hang can't take down the rest), and its result is measured: volume, tight
bounding box, solid count, validity. Models needing cadgen features beyond modeling
(assemblies, drawings) are reported as skipped on both backends.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

STANDIN = {
    "cadgen/__init__.py": '''
        import os as _os
        _REGISTRY = []

        def _decorator(kind):
            def deco(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    _REGISTRY.append((kind, args[0].__name__, args[0]))
                    return args[0]
                def wrap(fn):
                    _REGISTRY.append((kind, fn.__name__, fn))
                    return fn
                return wrap
            return deco

        step = _decorator("step")
        glb = _decorator("glb")
        stl = _decorator("stl")
        threemf = _decorator("threemf")
        dxf = _decorator("dxf")

        def srgb(value, alpha=1.0):
            v = value.lstrip("#")
            if len(v) == 3:
                v = "".join(c * 2 for c in v)
            def lin(c):
                return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            return tuple(lin(int(v[i:i + 2], 16) / 255) for i in (0, 2, 4)) + (alpha,)
    ''',
    "cadgen/build123d.py": '''
        import os as _os
        if _os.environ["T2C_BACKEND"] == "keel":
            from keel.build123d import *  # noqa: F401,F403
            import keel.build123d as _impl
        else:
            from build123d import *  # noqa: F401,F403
            import build123d as _impl

        def __getattr__(name):
            return getattr(_impl, name)
    ''',
    "cadgen/assembly.py": '''
        raise ImportError("harness: cadgen.assembly (mates/joints) is not part of this comparison")
    ''',
    "cadgen/eng_drawing.py": '''
        raise ImportError("harness: cadgen.eng_drawing is not part of this comparison")
    ''',
}

RUNNER = r'''
import importlib, json, math, os, pathlib, sys, time, traceback
module_name, fn_name = sys.argv[1], sys.argv[2]
out = {"model": module_name, "function": fn_name}
t0 = time.perf_counter()
try:
    import cadgen
    mod = importlib.import_module(module_name)
    fn = next(f for kind, name, f in cadgen._REGISTRY if name == fn_name)
    shape = fn()
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["volume"] = float(shape.volume)
    bb = shape.bounding_box()
    out["bbox"] = [[bb.min.X, bb.min.Y, bb.min.Z], [bb.max.X, bb.max.Y, bb.max.Z]]
    try:
        out["solids"] = len(shape.solids())
    except Exception:
        out["solids"] = None
    try:
        out["valid"] = bool(shape.is_valid() if callable(shape.is_valid) else shape.is_valid)
    except Exception:
        out["valid"] = None
    out["status"] = "ok"
except ImportError as e:
    out["status"] = "skipped" if "harness:" in str(e) else "error"
    out["error"] = f"{type(e).__name__}: {e}"
except Exception as e:
    out["status"] = "error"
    out["error"] = f"{type(e).__name__}: {e}"
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if "/keel/" in f.filename or "/models/" in f.filename]
    out["where"] = [f"{pathlib.Path(f.filename).name}:{f.lineno} {f.line}" for f in frames[-3:]]
print("T2C_RESULT " + json.dumps(out))
'''

LISTER = r'''
import importlib, json, sys
import cadgen
try:
    importlib.import_module(sys.argv[1])
    print("T2C_LIST " + json.dumps(list(dict.fromkeys(name for kind, name, f in cadgen._REGISTRY))))
except ImportError as e:
    print("T2C_LIST " + json.dumps({"skipped" if "harness:" in str(e) else "error": str(e)}))
except Exception as e:
    print("T2C_LIST " + json.dumps({"error": f"{type(e).__name__}: {e}"}))
'''


def _write_standin(root: Path) -> None:
    for rel, body in STANDIN.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip())


def _run(code: str, args: list[str], env: dict, cwd: Path, timeout: float) -> tuple[str, str]:
    try:
        r = subprocess.run([sys.executable, "-c", code, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "", "timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["build123d", "keel"], required=True)
    ap.add_argument("--models", required=True, type=Path, help="directory of model .py files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="model module names to run")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    standin = Path(tempfile.mkdtemp(prefix="t2c-cadgen-"))
    _write_standin(standin)
    keel_src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, T2C_BACKEND=args.backend, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(standin), str(args.models), str(keel_src), env.get("PYTHONPATH", "")])

    results = []
    modules = sorted(p.stem for p in args.models.glob("*.py") if not p.name.startswith("_"))
    if args.only:
        modules = [m for m in modules if m in args.only]
    for module in modules:
        stdout, stderr = _run(LISTER, [module], env, args.models, args.timeout)
        line = next((l for l in stdout.splitlines() if l.startswith("T2C_LIST ")), None)
        listed = json.loads(line[9:]) if line else {"error": (stderr.strip().splitlines() or ["no output"])[-1]}
        if isinstance(listed, dict):
            status = "skipped" if "skipped" in listed else "error"
            results.append({"model": module, "function": None, "status": status, "error": next(iter(listed.values()))})
            print(f"{status:7} {module}: {next(iter(listed.values()))[:100]}", flush=True)
            continue
        for fn in listed:
            stdout, stderr = _run(RUNNER, [module, fn], env, args.models, args.timeout)
            line = next((l for l in stdout.splitlines() if l.startswith("T2C_RESULT ")), None)
            res = json.loads(line[11:]) if line else {"model": module, "function": fn, "status": "crash",
                                                      "error": (stderr.strip().splitlines() or ["no output"])[-1][:300]}
            results.append(res)
            detail = f"vol={res['volume']:.3f}" if res["status"] == "ok" else res.get("error", "")[:110]
            print(f"{res['status']:7} {module}.{fn}: {detail}", flush=True)
    args.out.write_text(json.dumps(results, indent=1))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("SUMMARY", counts)


if __name__ == "__main__":
    main()
