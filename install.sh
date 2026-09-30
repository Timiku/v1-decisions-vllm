#!/bin/bash
# Install the typed-decisions endpoints into an installed vLLM.
#
# What it installs:
#   1. The `decisions` package
#      (overlay/vllm/entrypoints/generate/decisions/) into the vLLM tree.
#   2. api-router-wire-v0.30.patch: three upstream files in
#      vllm/entrypoints/ (generate/api_router.py plus the two launcher
#      files), which register the routes, build the serving state, load
#      backend plugins and read VLLM_TYPED_DECISIONS_BACKEND.
#
# Environment overrides:
#   DECISIONS_SRC        directory containing this repo
#                        (default: the directory holding this script)
#   VLLM_SITE_PACKAGES   the site-packages directory holding the `vllm`
#                        package (default: derived from `import vllm`)
#
# Exit codes:
#   0  installed, or already installed (package refreshed either way)
#   1  any failure (patch dry-run, patch apply, marker missing, compile)
#
# Idempotent: if the wiring marker is already present the patch is
# skipped and only the package is refreshed, so code updates land by
# re-running the script.

set -u

SRC="${DECISIONS_SRC:-$(cd "$(dirname "$0")" && pwd)}"
WIRE="$SRC/api-router-wire-v0.30.patch"
PACKAGE_SRC="$SRC/overlay/vllm/entrypoints/generate/decisions"

SITE="${VLLM_SITE_PACKAGES:-$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))' | tail -n 1)}" \
    || { echo "[decisions-endpoint] cannot locate the vLLM install" >&2; exit 1; }
TARGET="$SITE/vllm/entrypoints/generate/api_router.py"
DEC="$SITE/vllm/entrypoints/generate/decisions"

fail() {
    echo "[decisions-endpoint] $1 — refusing install" >&2
    [ -f /tmp/decisions-endpoint.patch.log ] && tail -20 /tmp/decisions-endpoint.patch.log >&2
    exit 1
}

[ -f "$WIRE" ] || fail "wire patch not found at $WIRE"
[ -d "$PACKAGE_SRC" ] || fail "overlay package not found at $PACKAGE_SRC"

# 1) overlay the package (recursive, fresh copy so updates land)
rm -rf "$DEC"
cp -r "$PACKAGE_SRC" "$DEC" || fail "copying the overlay package"
find "$DEC" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

# 2) wire the routes: skip the patch when the marker is already present
if grep -q "register_decisions_api_router" "$TARGET" 2>/dev/null; then
    echo "[decisions-endpoint] wiring already applied — package refreshed" >&2
else
    ( cd "$SITE" && patch -p1 --forward --batch --dry-run \
        < "$WIRE" > /tmp/decisions-endpoint.patch.log 2>&1 ) \
        || fail "wire patch dry-run failed"
    ( cd "$SITE" && patch -p1 --forward --batch \
        < "$WIRE" >> /tmp/decisions-endpoint.patch.log 2>&1 ) \
        || fail "wire patch failed to apply"
    grep -q "register_decisions_api_router" "$TARGET" \
        || fail "patch applied but the decisions marker is missing"
    grep -q "register_systemone_api_router" "$TARGET" \
        || fail "patch applied but the systemone marker is missing"
fi

# 3) compile-check the package and the patched files
python3 -m compileall -q "$DEC" \
    || fail "compileall failed on the decisions package"
python3 -m py_compile \
    "$TARGET" \
    "$SITE/vllm/entrypoints/launchers/api_server/routers.py" \
    "$SITE/vllm/entrypoints/launchers/api_server/app_state.py" \
    || fail "py_compile failed on the patched upstream files"

echo "[decisions-endpoint] installed: POST /v1/decisions + POST /v1/systemone" >&2
