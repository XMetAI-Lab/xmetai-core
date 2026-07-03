#!/usr/bin/env bash
# Disable torch-harmonics CUDA extension builds (EarthSpectralConv2d only needs SHT).
# Usage:
#   source scripts/install_torch_harmonics.sh
#   install_torch_harmonics
set -euo pipefail

# These env-vars tell torch-harmonics / torch.utils.cpp_extension not to compile CUDA ops.
export FORCE_CUDA_EXTENSION=0
export CUDA_VISIBLE_DEVICES=""
unset CUDA_HOME CUDA_PATH 2>/dev/null || true

install_torch_harmonics() {
  pip install "torch-harmonics==0.8.0" --no-build-isolation --no-cache-dir
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  install_torch_harmonics
fi
