#!/usr/bin/env bash
# SubTranslate Setup Script
# Sets up Python venv, installs PyTorch (ROCm / CUDA / CPU), Whisper, and all dependencies
#
# Environment overrides:
#   TORCH_BACKEND=rocm|cuda|cpu   Force a PyTorch build (default: auto-detect)
#   WHISPER_MODEL=large-v3        Whisper model to pre-download (set to "none" to skip)
#   OLLAMA_MODEL=qwen2.5:14b      Ollama model to pull

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║           SubTranslate — Setup                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check prerequisites ─────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 not found. Install Python 3.10 or newer."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "✗ Python 3.10+ is required (found $(python3 --version))"
    exit 1
fi
if ! command -v ffmpeg &> /dev/null || ! command -v ffprobe &> /dev/null; then
    echo "✗ FFmpeg (ffmpeg + ffprobe) is not installed!"
    echo "  Install it: sudo apt install ffmpeg / sudo pacman -S ffmpeg / sudo dnf install ffmpeg"
    exit 1
fi
echo "✓ Python $(python3 -c 'import platform; print(platform.python_version())') and FFmpeg found"

# ── Create directories ──────────────────────────────────────
mkdir -p INPUT OUTPUT
echo "✓ Created INPUT/ and OUTPUT/ directories"

# ── Create virtual environment ──────────────────────────────
if [ ! -d "venv" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip wheel setuptools
echo "✓ pip/wheel/setuptools upgraded"

# ── Detect GPU backend ─────────────────────────────────────
detect_backend() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        echo "cuda"
    elif command -v rocminfo &> /dev/null && rocminfo 2>/dev/null | grep -q "gfx"; then
        echo "rocm"
    else
        echo "cpu"
    fi
}
TORCH_BACKEND="${TORCH_BACKEND:-$(detect_backend)}"

case "$TORCH_BACKEND" in
    rocm) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3" ;;
    cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *)    echo "✗ Unknown TORCH_BACKEND '$TORCH_BACKEND' (use rocm, cuda or cpu)"; exit 1 ;;
esac

# ROCm consumer cards often need a GFX override (see run.sh for the same logic)
if [ "$TORCH_BACKEND" = "rocm" ] && [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
    GFX="$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]*' || true)"
    case "$GFX" in
        gfx103*) export HSA_OVERRIDE_GFX_VERSION=10.3.0 ;;
        gfx110*) export HSA_OVERRIDE_GFX_VERSION=11.0.0 ;;
    esac
fi

# ── Install PyTorch ─────────────────────────────────────────
echo ""
echo "→ Installing PyTorch ($TORCH_BACKEND build)..."
echo "  (This may take a while — downloading up to ~3GB)"
echo ""

if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch already installed ($(python3 -c 'import torch; print(torch.__version__)'))"
else
    pip install --no-cache-dir torch torchaudio --index-url "$TORCH_INDEX"
    echo "✓ PyTorch installed"
fi

# ── Verify GPU detection ───────────────────────────────────
echo ""
echo "→ Verifying GPU detection..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('  ⚠ No GPU detected — Whisper will run on CPU (slower)')
    print('  Tip: install ROCm/CUDA drivers, or force a build with TORCH_BACKEND=rocm|cuda')
"

# ── Install Python dependencies ─────────────────────────────
echo ""
echo "→ Installing Whisper and subtitle libraries..."
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"

# ── Download Whisper model ──────────────────────────────────
if [ "$WHISPER_MODEL" != "none" ]; then
    echo ""
    echo "→ Pre-downloading Whisper '$WHISPER_MODEL' model..."
    echo "  (large-v3 is ~3GB on first run; set WHISPER_MODEL=none to skip)"
    python3 -c "
import os, whisper
root = os.path.join(os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')), 'whisper')
whisper._download(whisper._MODELS['$WHISPER_MODEL'], root, False)
print('  ✓ Model downloaded and cached')
" || echo "  ⚠ Could not pre-download '$WHISPER_MODEL' (it will be downloaded on first run)"
fi

# ── Verify Ollama ───────────────────────────────────────────
echo ""
echo "→ Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama is installed"
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$OLLAMA_MODEL"; then
        echo "  ✓ $OLLAMA_MODEL model is available"
    else
        echo "  ⚠ $OLLAMA_MODEL not found. Pulling it now..."
        ollama pull "$OLLAMA_MODEL" || echo "  ⚠ Pull failed — is the Ollama service running? (ollama serve)"
    fi
else
    echo "  ⚠ Ollama is not installed — translation will not work until it is."
    echo "    Install it from: https://ollama.com/download"
    echo "    Transcription-only mode (--transcribe-only) works without it."
fi

chmod +x run.sh 2>/dev/null || true

# ── Done ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Setup Complete!                            ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Usage:                                              ║"
echo "║    1. Place files in INPUT/                          ║"
echo "║    2. Run: ./run.sh                                  ║"
echo "╚══════════════════════════════════════════════════════╝"
