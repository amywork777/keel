"""Access to NVIDIA's usd-brep (SMLib) kernel.

The kernel ships wheels for Linux and Windows only. On macOS run keel inside the
provided Docker image (``bin/keel script.py``).
"""

from __future__ import annotations

import contextlib
import os
import sys


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


try:
    import usd_brep as ub
except ImportError as exc:  # pragma: no cover - exercised only without the kernel
    hint = (
        "the usd-brep kernel has no macOS build; run your script with `bin/keel script.py` "
        "(Docker) instead of python directly"
        if sys.platform == "darwin"
        else "install it with `pip install usd-brep`"
    )
    raise ImportError(f"keel needs NVIDIA's usd-brep kernel: {hint}") from exc


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
    if _usd_warmed:
        return call(op, fn, *args, **kwargs)
    with _mute_c_stderr():
        result = call(op, fn, *args, **kwargs)
    _usd_warmed = True
    return result
