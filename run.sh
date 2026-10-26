#!/usr/bin/env bash
# SubTranslate — Run Script
# Convenience wrapper that activates venv, sets env vars, and runs the pipeline
#
#   ./run.sh                          Process everything in INPUT/
#   ./run.sh episode01.mkv [flags]    Process INPUT/episode01.mkv (or any path) with optional flags
#   ./run.sh --whisper-model medium   Pass flags straight through

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check venv exists ──
if [ ! -d "venv" ]; then
    echo "✗ Virtual environment not found. Run setup first:"
    echo "  bash setup.sh"
    exit 1
fi

# ── Activate venv ──
# shellcheck disable=SC1091
source venv/bin/activate

# ── Set AMD ROCm environment ──
# GFX override is hardware-specific: RDNA2 (RX 6000) uses 10.3.0, RDNA3 (RX 7000) uses 11.0.0.
# A value you export yourself always wins.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] && command -v rocminfo &> /dev/null; then
    GFX="$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)"
    case "$GFX" in
        gfx103*) export HSA_OVERRIDE_GFX_VERSION=10.3.0 ;;
        gfx110*) export HSA_OVERRIDE_GFX_VERSION=11.0.0 ;;
    esac
fi
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Run ──
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    # First argument is a filename: accept a real path, or a name inside INPUT/
    target="$1"
    shift
    if [ ! -e "$target" ] && [ -e "INPUT/$target" ]; then
        target="INPUT/$target"
    fi
    exec python3 subtranslate.py --file "$target" "$@"
else
    exec python3 subtranslate.py "$@"
fi
