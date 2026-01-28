"""
Pytest configuration and shared fixtures for xmetai tests.
Provides device management, data generation, and model initialization utilities.
"""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Get available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for CPU-only tests."""
    return torch.device("cpu")


@pytest.fixture(scope="function")
def torch_seed():
    """Seed for reproducibility in tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="function")
def sample_input_2d(torch_seed):
    """Generate sample 2D input tensor (B, L, C)."""
    B, L, C = 2, 256, 64  # batch_size, seq_length, hidden_dim
    return torch.randn(B, L, C)


@pytest.fixture(scope="function")
def sample_input_3d(torch_seed):
    """Generate sample 3D input tensor (B, H, W, C)."""
    B, H, W, C = 2, 16, 16, 64
    return torch.randn(B, H, W, C)


@pytest.fixture(scope="function")
def sample_input_4d(torch_seed):
    """Generate sample 4D input tensor for weather data (B, T, C, H, W)."""
    B, T, C, H, W = 2, 4, 65, 32, 64
    return torch.randn(B, T, C, H, W)


@pytest.fixture(scope="function")
def sample_freqs_cis(torch_seed):
    """Generate precomputed RoPE frequencies."""
    dim = 8  # match head_dim when C=64 and num_heads=8
    seq_len = 256
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


@pytest.fixture(scope="function")
def sample_weather_data(torch_seed):
    """Generate realistic weather data sample.

    Returns:
        dict with keys: 'input', 'target', 'mean', 'std'
    """
    B, T_in, C, H, W = 1, 2, 65, 32, 64  # 1 sample, 2 input timesteps

    return {
        "input": torch.randn(B, T_in, C, H, W),
        "target": torch.randn(B, 1, C, H, W),  # 1 output step
        "mean": torch.zeros(C, 1, 1),
        "std": torch.ones(C, 1, 1),
    }


def pytest_configure(config):
    """Configure pytest and register custom markers."""
    config.addinivalue_line("markers", "cpu: run only on CPU")
    config.addinivalue_line("markers", "gpu: run only on GPU (requires CUDA)")
    config.addinivalue_line("markers", "slow: slow integration tests")
    config.addinivalue_line("markers", "numerical: numerical stability tests")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
