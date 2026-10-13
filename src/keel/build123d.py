"""The old name for :mod:`keel123d` (``from keel import build123d`` still works)."""

import sys

import keel123d

sys.modules[__name__] = keel123d
