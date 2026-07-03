#!/usr/bin/env bash
set -euo pipefail

# EarthSpectralConv2d only needs RealSHT/InverseRealSHT (pure PyTorch).
# Avoid compiling DISCO CUDA extensions when CUDA headers/toolkit are incomplete.
export FORCE_CUDA_EXTENSION=0
unset CUDA_HOME || true

python -m pip install "torch-harmonics==0.8.0" --no-build-isolation --no-cache-dir
