#!/usr/bin/env bash
# Build NVIDIA's usd-brep kernel and its Python module natively (macOS, or Linux without
# the wheel), so keel runs without Docker.
#
#   native/build.sh                     kernel + Python module, no USD (about 1.5 min)
#   native/build.sh --usd <prefix>      also NVIDIA's USD layer, against an OpenUSD install
#   native/build.sh --build-usd         build OpenUSD 25.11 first (core only, ~15 min), then all
#
# Needs: cmake, ninja, a C++17 compiler, Python 3.12 (uv is used to make one if missing).
# Output: native/build/_omni_solid*.so, which keel picks up automatically.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEPS="${KEEL_DEPS:-$ROOT/.deps}"
USD_BREP_TAG="${USD_BREP_TAG:-main}"   # NVIDIA has no release tags yet; pin a commit with USD_BREP_REF
USD_TAG="v25.11"                        # the OpenUSD release NVIDIA builds against
USD_PREFIX=""
BUILD_USD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --usd) USD_PREFIX="$2"; shift 2 ;;
    --build-usd) BUILD_USD=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$DEPS"
SRC="${USD_BREP_SRC:-$DEPS/usd-brep}"
if [[ ! -d "$SRC/source/SMLib" ]]; then
  echo "== cloning NVIDIA-Omniverse/usd-brep"
  git clone -q --depth 1 --branch "$USD_BREP_TAG" https://github.com/NVIDIA-Omniverse/usd-brep.git "$SRC"
fi
if [[ -n "${USD_BREP_REF:-}" ]]; then
  git -C "$SRC" fetch -q --depth 1 origin "$USD_BREP_REF" && git -C "$SRC" checkout -q FETCH_HEAD
fi

if [[ $BUILD_USD -eq 1 ]]; then
  USD_PREFIX="${USD_PREFIX:-$DEPS/usd-install}"
  if [[ ! -f "$USD_PREFIX/pxrConfig.cmake" ]]; then
    [[ -d "$DEPS/OpenUSD" ]] || git clone -q --depth 1 --branch "$USD_TAG" \
      https://github.com/PixarAnimationStudios/OpenUSD.git "$DEPS/OpenUSD"
    echo "== building OpenUSD $USD_TAG (core libraries only) into $USD_PREFIX"
    python3 "$DEPS/OpenUSD/build_scripts/build_usd.py" --no-python --no-imaging --no-examples \
      --no-tutorials --no-tools --no-docs --no-tests --no-materialx --no-usdview \
      --build-variant release "$USD_PREFIX"
  fi
fi

PY="${KEEL_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "== making a Python 3.12 environment in .venv"
    uv venv -q -p 3.12 "$ROOT/.venv"
  fi
  PY="$ROOT/.venv/bin/python"
fi
"$PY" -m pybind11 --version >/dev/null 2>&1 || uv pip install -q -p "$PY" pybind11 pytest

echo "== configuring"
cmake -S "$HERE" -B "$HERE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSD_BREP_SRC="$SRC" -DPython_EXECUTABLE="$PY" ${USD_PREFIX:+-DUSD_ROOT="$USD_PREFIX"} >/dev/null
echo "== building"
ninja -C "$HERE/build"
echo "== done: $(ls "$HERE"/build/_omni_solid*.so)"
PYTHONPATH="$ROOT/src" "$PY" -c "import keel._kernel as k; print('keel kernel:', k.BACKEND, '| USD:', k.USD_AVAILABLE)"
