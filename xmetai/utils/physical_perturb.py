import math

import torch
import torch.nn.functional as F


def physical_input_perturb(
    input: torch.Tensor, weight: torch.Tensor, names: list[str], height_rms: float = 5.0, latitude_ascending: bool = True
) -> torch.Tensor:
    """Add a smooth barotropic Z/U/V initial perturbation in physical units.

    Input shape is (batch, history, channels, 721, 1440). Geopotential channels
    use m2/s2 and winds use m/s. ``height_rms`` is the global area-weighted RMS
    in geopotential-height metres. Latitude weights must be nonnegative.

    Independent members share their own perturbation across history frames and
    50--850 hPa levels. Winds follow discrete geostrophic increments derived
    from the same tapered potential; no second taper is applied to the winds.
    The smooth taper suppresses the tropics and polar caps. This is a limited
    initial-condition baseline, not a full balanced analysis perturbation.
    """
    if not math.isfinite(height_rms) or height_rms < 0:
        raise ValueError("height_rms must be finite and nonnegative (metres).")
    b, _, c, h, w = input.shape
    if (h, w) != (721, 1440):
        raise ValueError("physical_input_perturb requires a global 721x1440 grid.")
    if len(names) != c or len(set(names)) != c:
        raise ValueError("names must contain one unique name for each input channel.")
    levels = [int(n[1:]) for n in names if n.startswith("z") and n[1:].isdigit() and 50 <= int(n[1:]) <= 850]
    if not levels:
        raise ValueError("physical_input_perturb requires pressure-level Z/U/V channels.")
    indices = [[names.index(f"{var}{level}") for level in levels] for var in ("z", "u", "v")]
    with torch.autocast(device_type=input.device.type, enabled=False):
        lat = torch.linspace(-math.pi / 2, math.pi / 2, h, device=input.device)
        if not latitude_ascending:
            lat = lat.flip(0)
        deg = lat.abs() * (180 / math.pi)
        equatorial_ramp = ((deg - 20) / 15).clamp(0, 1)
        polar_ramp = ((80 - deg) / 10).clamp(0, 1)
        taper = (0.25 * (1 - (math.pi * equatorial_ramp).cos()) * (1 - (math.pi * polar_ramp).cos()))[:, None]
        weights = weight.to(device=input.device, dtype=torch.float32).reshape(h, -1).expand(h, w)
        if not torch.isfinite(weights).all() or (weights < 0).any() or (taper * weights).sum() <= 0:
            raise ValueError("weight must be finite, nonnegative and positive in the perturbation domain.")
        noise = torch.randn(b, 1, h, w, device=input.device, dtype=torch.float32)
        if not latitude_ascending:
            noise = noise.flip(-2)
        grid = torch.arange(-24, 25, device=input.device, dtype=torch.float32)
        kernel = (-0.5 * (grid / 8).square()).exp()
        kernel = kernel / kernel.sum()
        field = F.conv2d(F.pad(noise, (24, 24, 0, 0), mode="circular"), kernel.view(1, 1, 1, -1))
        field = F.conv2d(F.pad(field, (0, 0, 24, 24), mode="reflect"), kernel.view(1, 1, -1, 1))
        center = (field * taper * weights).sum((-2, -1), keepdim=True) / (taper * weights).sum()
        phi = (field - center) * taper
        rms = ((phi.square() * weights).sum((-2, -1), keepdim=True) / weights.sum()).sqrt()
        phi = phi * (height_rms * 9.80665 / rms.clamp_min(1e-12))
        dlat, dlon = math.pi / (h - 1), 2 * math.pi / w
        if not latitude_ascending:
            dlat = -dlat
        dy = (
            torch.cat(
                (
                    (phi[..., 1:2, :] - phi[..., :1, :]) / dlat,
                    (phi[..., 2:, :] - phi[..., :-2, :]) / (2 * dlat),
                    (phi[..., -1:, :] - phi[..., -2:-1, :]) / dlat,
                ),
                dim=-2,
            )
            / 6371000
        )
        cos = lat.cos().abs().clamp_min(math.cos(math.radians(80)))[:, None]
        dx = (phi.roll(-1, -1) - phi.roll(1, -1)) / (2 * dlon * 6371000 * cos)
        coriolis = torch.where(deg >= 20, 2 * 7.292115e-5 * lat.sin(), torch.ones_like(lat))[:, None]
        increments = (phi, -dy / coriolis, dx / coriolis)
        result = input.float().clone()
        for channels, delta in zip(indices, increments):
            result[:, :, channels] += delta.unsqueeze(1)
    return result
