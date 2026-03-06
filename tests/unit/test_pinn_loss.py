"""
Unit tests for physics-informed (PINN) loss functions.
Tests: ChannelRegistry, finite-difference helpers, all 6 PINN losses.
"""

import math

import pytest
import torch

from xmetai.loss.pinn_loss import (
    GRAVITY,
    LEVELS_13,
    PL_NAMES,
    ChannelRegistry,
    _coriolis,
    _dfdx,
    _dfdy,
    _make_lat_lon,
    energy_conservation_loss,
    geostrophic_balance_loss,
    hydrostatic_balance_loss,
    mass_continuity_loss,
    moisture_conservation_loss,
    pinn_loss,
    thermal_wind_loss,
)

# ---- Fixtures ----------------------------------------------------------------

S2S_CHANNELS = []
for var in PL_NAMES:
    for lvl in LEVELS_13:
        S2S_CHANNELS.append(f"{var}{lvl}")
SURFACE_NAMES = ["msl", "t2m", "sst", "ttr", "tp"]
S2S_CHANNELS += SURFACE_NAMES
# total: 5*13 + 5 = 70 channels (simplified s2s-like layout)

B, T, C, H, W = 2, 1, len(S2S_CHANNELS), 32, 64


@pytest.fixture
def registry():
    return ChannelRegistry(S2S_CHANNELS)


@pytest.fixture
def tensors():
    torch.manual_seed(42)
    output = torch.randn(B, T, C, H, W)
    target = torch.randn(B, T, C, H, W)
    return output, target


@pytest.fixture
def weight():
    lat = torch.linspace(90, -90, H)
    w = torch.cos(lat * math.pi / 180.0)
    return w.view(1, H, 1)


# ---- ChannelRegistry ---------------------------------------------------------


class TestChannelRegistry:
    def test_all_vars_found(self, registry):
        for var in PL_NAMES:
            assert registry.has_var(var)
            assert len(registry.indices(var)) == len(LEVELS_13)

    def test_indices_order(self, registry):
        z_indices = registry.indices("z")
        assert z_indices[0] == S2S_CHANNELS.index("z1000")
        assert z_indices[-1] == S2S_CHANNELS.index("z50")

    def test_pressure_pa(self, registry):
        p = registry.pressure_pa()
        assert p.shape == (len(LEVELS_13),)
        assert p[0].item() == pytest.approx(1000 * 100.0)
        assert p[-1].item() == pytest.approx(50 * 100.0)

    def test_missing_var(self):
        reg = ChannelRegistry(["z500", "t500"], levels=[500], pl_names=["z", "t", "u"])
        assert reg.has_var("z")
        assert reg.has_var("t")
        assert not reg.has_var("u")

    def test_partial_levels(self):
        reg = ChannelRegistry(["z500", "z850"], levels=[500, 850, 1000])
        assert not reg.has_var("z")
        assert len(reg.indices("z")) == 2


# ---- Grid Helpers ------------------------------------------------------------


class TestGridHelpers:
    def test_lat_lon_shape(self):
        lat, lon = _make_lat_lon(H, W, torch.device("cpu"))
        assert lat.shape == (H,)
        assert lon.shape == (W,)
        assert lat[0].item() == pytest.approx(math.pi / 2, abs=1e-5)
        assert lat[-1].item() == pytest.approx(-math.pi / 2, abs=1e-5)

    def test_dfdx_constant_field(self):
        lat, _ = _make_lat_lon(H, W, torch.device("cpu"))
        f = torch.ones(1, 1, H, W)
        dlon = 2 * math.pi / W
        result = _dfdx(f, lat, dlon)
        assert result.shape == f.shape
        assert result.abs().max().item() < 1e-5

    def test_dfdy_constant_field(self):
        f = torch.ones(1, 1, H, W)
        dlat = math.pi / (H - 1)
        result = _dfdy(f, dlat)
        assert result.shape == (1, 1, H - 2, W)
        assert result.abs().max().item() < 1e-5

    def test_dfdx_periodic(self):
        lat, _ = _make_lat_lon(H, W, torch.device("cpu"))
        dlon = 2 * math.pi / W
        lon = torch.linspace(0, 2 * math.pi, W + 1)[:-1]
        f = torch.sin(lon).view(1, 1, 1, W).expand(1, 1, H, W)
        result = _dfdx(f, lat, dlon)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_coriolis(self):
        lat = torch.tensor([math.pi / 2, 0.0, -math.pi / 2])
        f = _coriolis(lat)
        assert f[0].item() > 0
        assert f[1].item() >= 1e-5 - 1e-7  # clamped near equator (float32 tolerance)
        assert f[2].item() < 0 or f[2].item() >= 1e-5 - 1e-7  # south pole: sin(-90)=-1, clamped


# ---- Loss Functions ----------------------------------------------------------


class TestMassContinuityLoss:
    def test_output_scalar(self, tensors, registry):
        output, _ = tensors
        loss = mass_continuity_loss(output, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_zero_wind_zero_loss(self, registry):
        output = torch.zeros(B, T, C, H, W)
        loss = mass_continuity_loss(output, registry)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        loss = mass_continuity_loss(output, registry)
        loss.backward()
        assert output.grad is not None
        assert output.grad.abs().sum() > 0

    def test_with_weight(self, registry, weight):
        output = torch.randn(B, T, C, H, W)
        loss_w = mass_continuity_loss(output, registry, weight=weight)
        loss_nw = mass_continuity_loss(output, registry)
        assert loss_w.item() != loss_nw.item()


class TestGeostrophicBalanceLoss:
    def test_output_scalar(self, tensors, registry):
        output, _ = tensors
        loss = geostrophic_balance_loss(output, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        loss = geostrophic_balance_loss(output, registry)
        loss.backward()
        assert output.grad is not None

    def test_numerical_stability(self, registry):
        output = torch.randn(B, T, C, H, W) * 1e3
        loss = geostrophic_balance_loss(output, registry)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestHydrostaticBalanceLoss:
    def test_output_scalar(self, tensors, registry):
        output, _ = tensors
        loss = hydrostatic_balance_loss(output, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        loss = hydrostatic_balance_loss(output, registry)
        loss.backward()
        assert output.grad is not None

    def test_consistent_atmosphere(self, registry):
        output = torch.zeros(B, T, C, H, W)
        loss = hydrostatic_balance_loss(output, registry)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)


class TestEnergyConservationLoss:
    def test_output_scalar(self, tensors, registry):
        output, target = tensors
        loss = energy_conservation_loss(output, target, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_identical_output_target(self, registry):
        field = torch.randn(B, T, C, H, W)
        loss = energy_conservation_loss(field, field, registry)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        target = torch.randn(B, T, C, H, W)
        loss = energy_conservation_loss(output, target, registry)
        loss.backward()
        assert output.grad is not None


class TestMoistureConservationLoss:
    def test_output_scalar(self, tensors, registry):
        output, target = tensors
        loss = moisture_conservation_loss(output, target, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        target = torch.randn(B, T, C, H, W)
        loss = moisture_conservation_loss(output, target, registry)
        loss.backward()
        assert output.grad is not None


class TestThermalWindLoss:
    def test_output_scalar(self, tensors, registry):
        output, _ = tensors
        loss = thermal_wind_loss(output, registry)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        loss = thermal_wind_loss(output, registry)
        loss.backward()
        assert output.grad is not None

    def test_numerical_stability(self, registry):
        output = torch.randn(B, T, C, H, W) * 1e3
        loss = thermal_wind_loss(output, registry)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ---- Combined PINN Loss -----------------------------------------------------


class TestCombinedPinnLoss:
    ALL_COEFF = {"mass": 1e-4, "geo": 1e-4, "hydro": 1e-4, "energy": 1e-4, "moisture": 1e-4, "thermal": 1e-4, "pv": 1e-4}

    def test_no_loss_without_coefficients(self, tensors, registry):
        """Default coefficients are 0 — no losses returned unless explicitly enabled."""
        output, target = tensors
        losses = pinn_loss(output, target, registry)
        assert losses == {}

    def test_returns_all_keys(self, tensors, registry):
        output, target = tensors
        losses = pinn_loss(output, target, registry, coefficients=self.ALL_COEFF)
        expected_keys = {"loss_mass", "loss_geo", "loss_hydro", "loss_energy", "loss_moisture", "loss_thermal", "loss_pv"}
        assert set(losses.keys()) == expected_keys

    def test_all_values_finite(self, tensors, registry):
        output, target = tensors
        losses = pinn_loss(output, target, registry, coefficients=self.ALL_COEFF)
        for k, v in losses.items():
            assert not torch.isnan(v), f"{k} is NaN"
            assert not torch.isinf(v), f"{k} is Inf"

    def test_coefficients(self, tensors, registry):
        output, target = tensors
        coeff = {"mass": 0.0, "geo": 1.0, "hydro": 0.0, "energy": 0.0, "moisture": 0.0, "thermal": 0.0, "pv": 0.0}
        losses = pinn_loss(output, target, registry, coefficients=coeff)
        assert "loss_geo" in losses
        assert "loss_mass" not in losses

    def test_with_weight(self, tensors, registry, weight):
        output, target = tensors
        losses = pinn_loss(output, target, registry, weight=weight, coefficients=self.ALL_COEFF)
        assert len(losses) == 7

    def test_gradient_flow_combined(self, registry):
        output = torch.randn(B, T, C, H, W, requires_grad=True)
        target = torch.randn(B, T, C, H, W)
        losses = pinn_loss(output, target, registry, coefficients=self.ALL_COEFF)
        total = sum(losses.values())
        total.backward()
        assert output.grad is not None
        assert output.grad.abs().sum() > 0

    def test_sum_reduction(self, tensors, registry):
        output, target = tensors
        losses_mean = pinn_loss(output, target, registry, reduction="mean", coefficients=self.ALL_COEFF)
        losses_sum = pinn_loss(output, target, registry, reduction="sum", coefficients=self.ALL_COEFF)
        for k in losses_mean:
            assert losses_sum[k].item() >= losses_mean[k].item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
