"""Access to NVIDIA's usd-brep (SMLib) kernel.

keel uses, in order:
1. the ``usd_brep`` package from pip (Linux and Windows wheels), or
2. a native build of the same Python module (``_omni_solid``) made from NVIDIA's source by
   ``native/build.sh``, which is how keel runs natively on macOS.

``KEEL_NATIVE_DIR`` points at a native build directory; the default is ``native/build``
next to this repo's ``src``.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def _mute_c_stderr():
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _load_native():
    native = Path(os.environ.get("KEEL_NATIVE_DIR", Path(__file__).resolve().parents[2] / "native" / "build"))
    if not any(native.glob("_omni_solid*.so")):
        return None
    schema = native / "omniSolid" / "resources"
    if schema.is_dir():
        os.environ.setdefault("OMNISOLID_PLUGIN_PATH", str(schema))
    sys.path.insert(0, str(native))
    try:
        module = importlib.import_module("_omni_solid")
    finally:
        sys.path.remove(str(native))
    if getattr(module.usd, "AVAILABLE", True) and hasattr(module.usd, "ensure_plugin_registered"):
        try:
            module.usd.ensure_plugin_registered()
        except RuntimeError:
            pass  # reported again by the first USD call
    return module


try:
    import usd_brep as ub

    BACKEND = "usd-brep wheel"
except ImportError as exc:  # no wheel for this platform: try a native build
    ub = _load_native()
    if ub is None:
        hint = (
            "build it natively with `native/build.sh`, or run your script with `bin/keel script.py` (Docker)"
            if sys.platform == "darwin"
            else "install it with `pip install usd-brep`, or build it with `native/build.sh`"
        )
        raise ImportError(f"keel needs NVIDIA's usd-brep kernel: {hint}") from exc
    BACKEND = "native build"

USD_AVAILABLE = getattr(ub.usd, "AVAILABLE", True)


class KeelError(RuntimeError):
    """A modeling operation failed in the kernel."""


def call(op: str, fn, *args, **kwargs):
    """Run a kernel function, re-raising kernel failures with the keel operation name."""
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise KeelError(f"{op} failed: {exc}") from exc


_usd_warmed = False


def usd_call(op: str, fn, *args, **kwargs):
    """Like ``call`` for USD file operations.

    The kernel's bundled OpenUSD tries to load the `pxr` Python modules (not shipped in the
    wheel) the first time it opens a stage and prints harmless warnings for each. Mute the
    C-level stderr for that first call; kernel failures still surface as KeelError.
    """
    global _usd_warmed
    if not USD_AVAILABLE:
        raise KeelError(
            f"{op} failed: this kernel build has no USD support (export .stl/.obj instead, "
            "or rebuild with OpenUSD: native/build.sh --usd <prefix>)"
        )
    if _usd_warmed:
        return call(op, fn, *args, **kwargs)
    with _mute_c_stderr():
        result = call(op, fn, *args, **kwargs)
    _usd_warmed = True
    return result


def usd_fn(name: str):
    """A function from the kernel's ``usd`` submodule, or a clear error if this build has none."""
    if not USD_AVAILABLE:
        raise KeelError(
            "this kernel build has no USD support (export .stl/.obj instead, "
            "or rebuild with OpenUSD: native/build.sh --usd <prefix>)"
        )
    return getattr(ub.usd, name)
