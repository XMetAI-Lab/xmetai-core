#!/usr/bin/env bash
#
# XMetAI Stage 1 environment check
#
# Read-only checks for:
#   - Python / virtual environment
#   - key Python packages
#   - PyTorch / CUDA / GPU
#   - Mamba CUDA extension
#   - distributed launcher environment
#   - XMetAI project / Git
#   - Stage 1 synthetic S2S dataset
#   - workspace storage
#
# Usage:
#   bash stage1_env_check.sh
#   bash stage1_env_check.sh --output /path/to/report.log
#

set -uo pipefail


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xmetai-core}"
VENV_ROOT="${XMETAI_VENV:-/workspace/venvs/xmetai}"
STAGE1_ROOT="${STAGE1_ROOT:-/workspace/stage1}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"

TOY_DATA="${TOY_DATA:-/workspace/data/xmetai-demo/s2s_toy.c8}"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-chore/stage1-onboarding}"

OUTPUT_FILE=""

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0


# ============================================================
# Helpers
# ============================================================

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

section() {
    echo
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo "[PASS] $*"
}

warn() {
    WARN_COUNT=$((WARN_COUNT + 1))
    echo "[WARN] $*"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[FAIL] $*"
}

info() {
    echo "[INFO] $*"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}


# ============================================================
# CLI
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --output requires a file path."
                exit 2
            fi
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage:
  $0 [--output FILE]

Options:
  --output FILE   Save the complete check output to FILE.
  -h, --help      Show this help message.
EOF
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            exit 2
            ;;
    esac
done


if [[ -n "$OUTPUT_FILE" ]]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    exec > >(tee "$OUTPUT_FILE") 2>&1
fi


# ============================================================
# Header
# ============================================================

echo "XMetAI Stage 1 Environment Check"
echo "Generated : $(timestamp)"
echo "Host      : $(hostname)"
echo "User      : $(whoami)"
echo "Project   : $PROJECT_ROOT"
echo "Venv      : $VENV_ROOT"
echo "Toy data  : $TOY_DATA"


# ============================================================
# 1. Basic system
# ============================================================

section "1. BASIC SYSTEM"

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    info "OS: ${PRETTY_NAME:-unknown}"
else
    warn "/etc/os-release not found."
fi

info "Kernel: $(uname -r)"
info "Architecture: $(uname -m)"

if command_exists nproc; then
    info "Visible CPUs: $(nproc)"
fi

if command_exists free; then
    free -h | sed 's/^/    /'
fi


# ============================================================
# 2. Python / virtual environment
# ============================================================

section "2. PYTHON / VIRTUAL ENVIRONMENT"

if [[ -x "$VENV_ROOT/bin/python" ]]; then
    PYTHON_BIN="$VENV_ROOT/bin/python"
    pass "Canonical XMetAI venv Python exists."
else
    fail "Venv Python not found: $VENV_ROOT/bin/python"

    if command_exists python; then
        PYTHON_BIN="$(command -v python)"
        warn "Falling back to: $PYTHON_BIN"
    else
        echo
        echo "Cannot continue without Python."
        exit 1
    fi
fi

CURRENT_PYTHON="$(command -v python 2>/dev/null || true)"
info "Shell python : ${CURRENT_PYTHON:-not found}"
info "Check python : $PYTHON_BIN"

PYTHON_EXEC="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
PYTHON_VER="$("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

info "sys.executable : $PYTHON_EXEC"
info "Python version : $PYTHON_VER"

if [[ "$PYTHON_EXEC" == "$VENV_ROOT/bin/python" ]]; then
    pass "Environment check is using the XMetAI venv interpreter."
else
    fail "Python interpreter is not the expected XMetAI venv interpreter."
fi

if [[ "$PYTHON_VER" == 3.11.* ]]; then
    pass "Python 3.11 baseline detected."
else
    warn "Stage 1 baseline used Python 3.11; current version is $PYTHON_VER."
fi


# ============================================================
# Internal runtime environment
# ============================================================

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VENV_ROOT/bin:$CUDA_HOME/bin:$PATH"

TORCH_LIB="$(
    "$PYTHON_BIN" - <<'PY'
import os
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    print("")
PY
)"

if [[ -n "$TORCH_LIB" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/configs:${PYTHONPATH:-}"


# ============================================================
# 3. Key Python packages
# ============================================================

section "3. KEY PYTHON PACKAGES"

check_module() {
    local label="$1"
    local module="$2"
    local expected="${3:-}"

    local output
    local status

    output="$(
        "$PYTHON_BIN" - "$module" <<'PY' 2>&1
import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

name = sys.argv[1]
m = importlib.import_module(name)

version = getattr(m, "__version__", "N/A")
path = getattr(m, "__file__", "N/A")

print(version)
print(path)
PY
    )"
    status=$?

    if [[ $status -ne 0 ]]; then
        fail "$label import failed."
        echo "$output" | sed 's/^/    /'
        return
    fi

    local version
    local path

    version="$(echo "$output" | sed -n '1p')"
    path="$(echo "$output" | sed -n '2p')"

    info "$label version : $version"
    info "$label path    : $path"

    if [[ -n "$expected" && "$version" != "$expected" ]]; then
        warn "$label version differs from Stage 1 baseline: expected $expected, got $version."
    else
        pass "$label import OK."
    fi
}


check_module "NumPy"       "numpy"       "1.25.0"
check_module "PyTorch"     "torch"       "2.11.0+cu128"
check_module "Xarray"      "xarray"      "2024.10.0"
check_module "Zarr"        "zarr"        "2.18.3"
check_module "Numcodecs"   "numcodecs"   "0.13.0"
check_module "Pytest"      "pytest"
check_module "mamba-ssm"   "mamba_ssm"   "2.2.5"

check_module "selective_scan_cuda" "selective_scan_cuda"
check_module "XMetAI" "xmetai"


# ============================================================
# 4. CUDA / GPU
# ============================================================

section "4. CUDA / GPU"

if command_exists nvcc; then
    pass "nvcc found: $(command -v nvcc)"
    nvcc --version | tail -n 4 | sed 's/^/    /'
else
    warn "nvcc not found in PATH."
fi

if command_exists nvidia-smi; then
    pass "nvidia-smi available."
    echo
    echo "Physical GPU / MIG inventory:"
    nvidia-smi -L | sed 's/^/    /'
else
    warn "nvidia-smi not found."
fi

CUDA_OUTPUT="$(
    "$PYTHON_BIN" - <<'PY' 2>&1
import sys
import torch

print("PyTorch           :", torch.__version__)
print("PyTorch CUDA      :", torch.version.cuda)
print("CUDA available    :", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit(1)

print("Device count      :", torch.cuda.device_count())
print("Device name       :", torch.cuda.get_device_name(0))
print("Capability        :", torch.cuda.get_device_capability(0))

props = torch.cuda.get_device_properties(0)

print(
    "Visible memory    :",
    f"{props.total_memory / 1024**3:.2f} GiB",
)

free, total = torch.cuda.mem_get_info()

print(
    "Free memory now   :",
    f"{free / 1024**3:.2f} GiB",
)
print(
    "Total CUDA memory :",
    f"{total / 1024**3:.2f} GiB",
)
PY
)"
CUDA_STATUS=$?

echo "$CUDA_OUTPUT" | sed 's/^/    /'

if [[ $CUDA_STATUS -eq 0 ]]; then
    pass "PyTorch CUDA runtime is operational."
else
    fail "PyTorch cannot use CUDA."
fi


# ============================================================
# 5. Mamba CUDA extension
# ============================================================

section "5. MAMBA CUDA EXTENSION"

MAMBA_OUTPUT="$(
    "$PYTHON_BIN" - <<'PY' 2>&1
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import mamba_ssm
import selective_scan_cuda

print("Python              :", sys.executable)
print("Torch               :", torch.__version__)
print("mamba_ssm           :", mamba_ssm.__file__)
print("selective_scan_cuda :", selective_scan_cuda.__file__)
PY
)"
MAMBA_STATUS=$?

echo "$MAMBA_OUTPUT" | sed 's/^/    /'

if [[ $MAMBA_STATUS -eq 0 ]]; then
    if echo "$MAMBA_OUTPUT" | grep -Fq "mamba_ssm           : $VENV_ROOT/" \
        && echo "$MAMBA_OUTPUT" | grep -Fq "selective_scan_cuda : $VENV_ROOT/"; then
        pass "Mamba and selective_scan_cuda both import from the XMetAI venv."
    else
        warn "Mamba imports successfully, but one or more module paths are outside the expected venv."
    fi
else
    fail "Mamba CUDA extension import failed."
fi


# ============================================================
# 6. Distributed launcher
# ============================================================

section "6. DISTRIBUTED LAUNCHER"

TORCHRUN_BIN="$(command -v torchrun 2>/dev/null || true)"

if [[ -n "$TORCHRUN_BIN" ]]; then
    info "torchrun in PATH: $TORCHRUN_BIN"

    TORCHRUN_SHEBANG="$(head -n 1 "$TORCHRUN_BIN" 2>/dev/null || true)"
    info "torchrun shebang: $TORCHRUN_SHEBANG"

    if [[ "$TORCHRUN_BIN" == "$VENV_ROOT/"* ]]; then
        pass "torchrun belongs to the XMetAI venv."
    else
        warn "torchrun is outside the XMetAI venv; prefer:"
        echo "    $VENV_ROOT/bin/python -m torch.distributed.run"
    fi
else
    warn "torchrun command not found; module launcher will be tested instead."
fi


TMP_DIST="$(mktemp /tmp/xmetai_stage1_dist_XXXXXX.py)"

cat > "$TMP_DIST" <<'PY'
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import selective_scan_cuda

print("child_python=" + sys.executable)
print("torch=" + torch.__version__)
print("selective_scan_cuda=" + selective_scan_cuda.__file__)
PY

DIST_OUTPUT="$(
    "$PYTHON_BIN" \
        -m torch.distributed.run \
        --standalone \
        --nproc_per_node=1 \
        "$TMP_DIST" \
        2>&1
)"
DIST_STATUS=$?

rm -f "$TMP_DIST"

echo "$DIST_OUTPUT" | sed 's/^/    /'

if [[ $DIST_STATUS -eq 0 ]] \
    && echo "$DIST_OUTPUT" | grep -q "child_python=$VENV_ROOT/bin/python" \
    && echo "$DIST_OUTPUT" | grep -q "$VENV_ROOT/lib/python3.11/site-packages/selective_scan_cuda"; then

    pass "Distributed child process uses the correct venv and Mamba extension."
else
    fail "Distributed launcher environment is incorrect."
fi


# ============================================================
# 7. XMetAI project / Git
# ============================================================

section "7. XMETAI PROJECT / GIT"

if [[ -d "$PROJECT_ROOT/.git" ]]; then
    pass "XMetAI Git repository found."
else
    fail "Git repository not found at $PROJECT_ROOT."
fi

if command_exists git && [[ -d "$PROJECT_ROOT/.git" ]]; then

    GIT_VERSION="$(git --version)"
    BRANCH="$(git -C "$PROJECT_ROOT" branch --show-current)"
    COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
    ORIGIN="$(git -C "$PROJECT_ROOT" remote get-url origin 2>/dev/null || true)"
    STATUS="$(git -C "$PROJECT_ROOT" status --porcelain)"

    info "$GIT_VERSION"
    info "Branch : $BRANCH"
    info "Commit : $COMMIT"
    info "Origin : ${ORIGIN:-not configured}"

    if [[ "$BRANCH" == "$EXPECTED_BRANCH" ]]; then
        pass "Expected Stage 1 branch is active."
    else
        warn "Expected branch '$EXPECTED_BRANCH', current branch is '$BRANCH'."
    fi

    if [[ -z "$STATUS" ]]; then
        pass "Git working tree is clean."
    else
        warn "Git working tree contains changes:"
        echo "$STATUS" | sed 's/^/    /'
    fi

else
    fail "Git command is unavailable."
fi


# ============================================================
# 8. Stage 1 toy S2S dataset
# ============================================================

section "8. STAGE 1 TOY S2S DATASET"

if [[ -d "$TOY_DATA" ]]; then
    pass "Toy Zarr directory exists."
else
    fail "Toy Zarr directory is missing: $TOY_DATA"
fi

for sidecar in mean.npy std.npy weight.npy const.npy; do
    if [[ -f "$TOY_DATA/$sidecar" ]]; then
        pass "Sidecar exists: $sidecar"
    else
        fail "Missing sidecar: $TOY_DATA/$sidecar"
    fi
done


DATA_OUTPUT="$(
    "$PYTHON_BIN" - "$TOY_DATA" <<'PY' 2>&1
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

path = sys.argv[1]

ds = xr.open_zarr(path, consolidated=True)

print("dims       :", dict(ds.sizes))
print("variables  :", list(ds.data_vars))
print("levels     :", ds.level.values.tolist())
print("time start :", ds.time.values[0])
print("time end   :", ds.time.values[-1])

for name in ["mean", "std", "weight", "const"]:
    arr = np.load(f"{path}/{name}.npy")
    print(f"{name:10s}: shape={arr.shape}, dtype={arr.dtype}")

expected_shape = (64, 8, 121, 240)

if tuple(ds["data"].shape) != expected_shape:
    raise RuntimeError(
        f"Unexpected toy data shape: {ds['data'].shape}, "
        f"expected {expected_shape}"
    )
PY
)"
DATA_STATUS=$?

echo "$DATA_OUTPUT" | sed 's/^/    /'

if [[ $DATA_STATUS -eq 0 ]]; then
    pass "Synthetic S2S Zarr can be opened and has the expected Stage 1 shape."
else
    fail "Synthetic S2S Zarr validation failed."
fi


# ============================================================
# 9. Workspace / storage
# ============================================================

section "9. WORKSPACE / STORAGE"

if [[ -d "$WORKSPACE_ROOT" ]]; then
    pass "Workspace exists: $WORKSPACE_ROOT"

    echo
    df -h "$WORKSPACE_ROOT" | sed 's/^/    /'
else
    fail "Workspace is missing: $WORKSPACE_ROOT"
fi

echo
info "Directory usage:"

for path in \
    "$PROJECT_ROOT" \
    "/workspace/data" \
    "$STAGE1_ROOT" \
    "/workspace/output"
do
    if [[ -e "$path" ]]; then
        du -sh "$path" 2>/dev/null | sed 's/^/    /'
    else
        echo "    $path : not present"
    fi
done


# ============================================================
# Summary
# ============================================================

section "SUMMARY"

echo "PASS : $PASS_COUNT"
echo "WARN : $WARN_COUNT"
echo "FAIL : $FAIL_COUNT"

echo

if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "STAGE1 ENVIRONMENT CHECK: SUCCESS"

    if [[ $WARN_COUNT -gt 0 ]]; then
        echo "Result: usable environment with non-fatal warnings."
    else
        echo "Result: all checks passed."
    fi

    EXIT_CODE=0
else
    echo "STAGE1 ENVIRONMENT CHECK: FAILED"
    echo "Result: one or more required checks failed."

    EXIT_CODE=1
fi

echo
echo "Finished: $(timestamp)"

exit "$EXIT_CODE"
