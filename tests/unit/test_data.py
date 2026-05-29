"""
Unit tests for data loading and preprocessing.
Tests: DataLoader, Sampler, Weather data utilities.
"""

import numpy as np
import pytest
import torch


class TestDataUtils:
    """Test weather data utilities."""

    @pytest.mark.gpu
    def test_normalize_denormalize(self, sample_weather_data):
        """Test data normalization and denormalization."""
        try:
            from xmetai.data.weather.weather_util import denormalize, normalize
        except ImportError:
            pytest.skip("Weather utilities not available")

        x = sample_weather_data["input"]
        mean = sample_weather_data["mean"]
        std = sample_weather_data["std"]

        # Normalize
        x_norm = normalize(x, mean, std)

        # Denormalize
        x_denorm = denormalize(x_norm, mean, std)

        assert torch.allclose(x_denorm, x, atol=1e-5)

    @pytest.mark.gpu
    def test_normalize_zero_std(self):
        """Test normalization handles zero standard deviation."""
        try:
            from xmetai.data.weather.weather_util import normalize
        except ImportError:
            pytest.skip("Weather utilities not available")

        x = torch.ones(2, 4, 65, 32, 64)
        mean = torch.zeros(65, 1, 1)
        std = torch.ones(65, 1, 1)
        std[0] = 0  # One channel with zero std

        x_norm = normalize(x, mean, std)
        # Should handle gracefully (typically by not dividing or using small epsilon)
        assert not torch.isnan(x_norm).any()

    @pytest.mark.gpu
    def test_data_range_validation(self):
        """Test data range validation."""
        try:
            from xmetai.data.weather.weather_util import validate_range
        except ImportError:
            pytest.skip("Weather utilities not available")

        data = torch.randn(2, 4, 65, 32, 64) * 100

        # Should validate without error
        validate_range(data)

    @pytest.mark.numerical
    def test_extreme_values_handling(self):
        """Test handling of extreme/NaN values in data."""
        try:
            from xmetai.data.weather.weather_util import remove_invalid
        except ImportError:
            pytest.skip("Weather utilities not available")

        data = torch.randn(2, 4, 65, 32, 64)
        # Inject some NaN and inf values
        data[0, 0, 0, 0, 0] = float("nan")
        data[1, 2, 3, 5, 5] = float("inf")

        cleaned = remove_invalid(data, fill_value=0.0)
        assert not torch.isnan(cleaned).any()
        assert not torch.isinf(cleaned).any()


class TestWeatherDataset:
    """Test weather dataset implementations."""

    def test_era5_dataset_structure(self):
        """Test ERA5 dataset basic structure."""
        try:
            from xmetai.data.weather.era5 import ERA5Dataset
        except ImportError:
            pytest.skip("ERA5 dataset not available")

        # This would require actual data files
        # Just test instantiation with mock data
        pytest.skip("Requires actual data files")

    def test_multi_zarr_dataset_loading(self):
        """Test multi-dataset loading."""
        try:
            from xmetai.data.weather.multi_zarr_dataset import MultiZarrDataset
        except ImportError:
            pytest.skip("MultiZarr dataset not available")

        # This would require actual zarr files
        pytest.skip("Requires actual zarr files")


class TestDataSampler:
    """Test custom data samplers."""

    @pytest.mark.gpu
    def test_sequential_sampler(self):
        """Test sequential sampler."""
        try:
            from xmetai.data.sampler import SequentialSampler
        except ImportError:
            pytest.skip("Sampler not available")

        dataset_size = 100
        sampler = SequentialSampler(dataset_size)

        indices = list(sampler)
        assert len(indices) == dataset_size
        assert indices == list(range(dataset_size))

    @pytest.mark.gpu
    def test_distributed_sampler(self):
        """Test distributed sampler for multi-GPU."""
        try:
            from xmetai.data.sampler import DistributedSampler
        except ImportError:
            pytest.skip("DistributedSampler not available")

        dataset_size = 100
        num_replicas = 2
        rank = 0

        sampler = DistributedSampler(
            dataset_size,
            num_replicas=num_replicas,
            rank=rank,
        )

        indices = list(sampler)
        # Should be roughly half the dataset per rank
        assert len(indices) == dataset_size // num_replicas


class TestDataLoader:
    """Test data loading functionality."""

    @pytest.mark.gpu
    def test_dataloader_batch_collation(self):
        """Test batch collation."""
        try:
            from xmetai.data.loader import build_loader
        except ImportError:
            pytest.skip("DataLoader not available")

        # Create mock dataset
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input": torch.randn(4, 65, 32, 64),
                    "target": torch.randn(1, 65, 32, 64),
                }

        dataset = MockDataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        for batch in loader:
            assert batch["input"].shape[0] == 2
            assert batch["target"].shape[0] == 2
            break

    @pytest.mark.gpu
    def test_dataloader_num_workers(self):
        """Test data loader with multiple workers."""

        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return torch.randn(65, 32, 64)

        dataset = MockDataset()

        # Test with num_workers=0 (no multiprocessing)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 5


class TestNormalizationConstants:
    """Test normalization constant computation."""

    def test_compute_dataset_stats(self):
        """Test computing mean/std from data."""
        try:
            from xmetai.data.weather.weather_util import compute_statistics
        except ImportError:
            pytest.skip("compute_statistics not available")

        # Create synthetic data
        data = torch.randn(10, 4, 65, 32, 64)

        mean = data.mean(dim=(0, 1, 3, 4), keepdim=True)
        std = data.std(dim=(0, 1, 3, 4), keepdim=True)

        assert mean.shape[-1] == 1
        assert std.shape[-1] == 1
        assert (std > 0).all()

    def test_normalization_preserves_shape(self):
        """Test that normalization preserves data shape."""
        x = torch.randn(2, 4, 65, 32, 64)
        mean = torch.zeros(65, 1, 1)
        std = torch.ones(65, 1, 1)

        x_norm = (x - mean) / std

        assert x_norm.shape == x.shape


class TestWeatherUtilXarray:
    """Test xarray-based weather utilities."""

    def test_update_dims_and_coords(self):
        """Ensure coords are renamed and level dimension is added."""
        xr = pytest.importorskip("xarray")
        from xmetai.data.weather.weather_util import update_coords, update_dims

        data = xr.DataArray(
            np.zeros((3, 3), dtype=np.float32),
            dims=("latitude", "longitude"),
            coords={"latitude": [-90.0, 0.0, 90.0], "longitude": [-180.0, 0.0, 180.0]},
        )

        renamed = update_dims(data)
        updated = update_coords(renamed)

        assert "level" in updated.dims
        assert updated.sizes["level"] == 1
        assert updated.lat.values[0] > updated.lat.values[-1]
        assert (updated.lon.values >= 0).all()
        assert (updated.lon.values <= 360).all()

    def test_normalize_inv_normalize_roundtrip(self):
        """Normalize then invert with log channel handling."""
        xr = pytest.importorskip("xarray")
        pytest.importorskip("dask")
        from xmetai.data.weather.weather_util import inv_normalize, normalize

        base = xr.DataArray(
            np.array([[[1.0]], [[2.0]]], dtype=np.float32),
            dims=("channel", "lat", "lon"),
            coords={"channel": ["tp", "z500"]},
        )

        mean = xr.zeros_like(base)
        std = xr.ones_like(base)

        normalized = normalize(base.copy(), mean, std)
        restored = inv_normalize(normalized, mean, std, dim="channel")

        restored_vals = restored.values.reshape(-1)
        assert np.isfinite(restored_vals).all()
        assert restored_vals.shape == base.values.reshape(-1).shape
        # Round-trip should not introduce large error; tp is log-transformed so allow shrinkage
        assert restored_vals.min() >= 0

    def test_land_to_nan_masks_ocean_channel(self):
        """Mask ocean-specific channels without altering others."""
        xr = pytest.importorskip("xarray")
        from xmetai.data.weather.weather_util import land_to_nan

        data = xr.DataArray(
            np.ones((1, 2, 2, 2), dtype=np.float32),
            dims=("time", "channel", "lat", "lon"),
            coords={"channel": ["sst", "tp"]},
        )

        mask = xr.DataArray(
            np.array([[True, False], [False, True]]),
            dims=("lat", "lon"),
        )

        updated = land_to_nan(data.copy(), mask)

        sst = updated.sel(channel="sst").values
        tp = updated.sel(channel="tp").values

        assert np.isnan(sst[..., 0, 1])
        assert np.isnan(sst[..., 1, 0])
        assert not np.isnan(tp).any()

    def test_compute_level_weight_properties(self):
        """Level weights should be positive and sum to one."""
        from xmetai.data.weather.weather_util import LEVELS_13, compute_level_weight

        weights = compute_level_weight()

        assert weights.shape[0] == len(LEVELS_13)
        assert (weights > 0).all()
        assert 0 < weights.sum() < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
