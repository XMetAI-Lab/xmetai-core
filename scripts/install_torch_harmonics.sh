#!/usr/bin/env bash
# Disable torch-harmonics CUDA extension builds (EarthSpectralConv2d only needs SHT).
# Usage:
#   source scripts/install_torch_harmonics.sh
#   install_torch_harmonics
set -euo pipefail

export FORCE_CUDA_EXTENSION=0
export CUDA_VISIBLE_DEVICES=""
unset CUDA_HOME CUDA_PATH || true

if command -v nvcc >/dev/null 2>&1; then
  _nvcc_dir="$(dirname "$(command -v nvcc)")"
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -Fv "$_nvcc_dir" | paste -sd: -)"
  export PATH
fi

install_torch_harmonics() {
  pip install "torch-harmonics==0.8.0" --no-build-isolation --no-cache-dir
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  install_torch_harmonics
fi
