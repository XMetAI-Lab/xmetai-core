"""GT-weighted Charbonnier loss: five weight levels, with single or double tails."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import torch
from torch import nn

__all__ = ["GTWeightedLoss"]

_FeatureSpec = tuple[int, int, tuple[float, ...], int, float, float, float, float]


@cache
def _load_triton_backend() -> type[torch.autograd.Function]:
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise ImportError("GTWeightedLoss requires Triton installed in the CUDA environment") from exc
    from torch.autograd.function import once_differentiable

    @triton.jit
    def _offset(n: tl.tensor, channel: tl.tensor, shape: tl.constexpr, strides: tl.constexpr) -> tl.tensor:
        # Rollout slices retain their parent strides, which can exceed int32.
        n = n.to(tl.int64)
        channel = channel.to(tl.int64)
        t: tl.constexpr = shape[1]
        h: tl.constexpr = shape[2]
        w: tl.constexpr = shape[3]
        return n // (t * h * w) * strides[0] + n // (h * w) % t * strides[1] + channel * strides[2] + n // w % h * strides[3] + n % w * strides[4]

    @triton.jit
    def _pick(feature: tl.tensor, table: tl.constexpr, column: tl.constexpr, dtype: tl.constexpr = tl.float32) -> tl.tensor:
        # Only small scalar metadata is selected; the point computation is not unrolled.
        # Keep tuple slicing inside the helper: Triton 3.6 does not normalize a
        # raw nested tuple passed directly as a JIT argument.
        values: tl.constexpr = table[column]
        result = tl.full((), values[0], dtype)
        for i in tl.static_range(1, len(values)):
            result = tl.where(feature == i, values[i], result)
        return result

    @triton.jit
    def _component(
        Output: tl.tensor,
        Weight: tl.tensor,
        Slope: tl.tensor,
        n: tl.tensor,
        channel: tl.tensor,
        slot: tl.tensor,
        truth: tl.tensor,
        valid: tl.tensor,
        bins: tl.tensor,
        N: tl.constexpr,
        SHAPE: tl.constexpr,
        OS: tl.constexpr,
        WS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        SAVE_SLOPE: tl.constexpr,
        EPS: tl.constexpr,
    ) -> tl.tensor:
        prediction = tl.load(Output + _offset(n, channel, SHAPE, OS), n < N, other=0).to(tl.float32)
        multiplier = bins
        if HAS_WEIGHT:
            multiplier *= tl.load(Weight + _offset(n, channel, SHAPE, WS), n < N, other=0).to(tl.float32)
        error = tl.where(valid, prediction, 0.0) - tl.where(valid, truth, 0.0)
        root = tl.sqrt(error * error + EPS)
        if SAVE_SLOPE:
            tl.store(Slope + slot.to(tl.int64) * N + n, tl.where(valid, error / root * multiplier, 0.0), n < N)
        return tl.where(valid, (root - tl.sqrt(EPS)) * multiplier, 0.0)

    @triton.jit
    def _gt_forward(
        Output: tl.tensor,
        Target: tl.tensor,
        Weight: tl.tensor,
        Mask: tl.tensor,
        Partial: tl.tensor,
        Counts: tl.tensor,
        Slope: tl.tensor,
        N: tl.constexpr,
        SHAPE: tl.constexpr,
        STRIDES: tl.constexpr,
        META: tl.constexpr,
        CUTOFFS: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        HAS_MASK: tl.constexpr,
        SAVE_SLOPE: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        feature = tl.program_id(1)
        os: tl.constexpr = STRIDES[0]
        ts: tl.constexpr = STRIDES[1]
        ws: tl.constexpr = STRIDES[2]
        ms: tl.constexpr = STRIDES[3]
        # Metadata columns: source u/v, compact u/v, lower count, mean/std u/v, cutoff count.
        u = _pick(feature, META, 0, tl.int32)
        v = _pick(feature, META, 1, tl.int32)
        slot_u = _pick(feature, META, 2, tl.int32)
        slot_v = _pick(feature, META, 3, tl.int32)
        lower = _pick(feature, META, 4, tl.int32)
        cutoff_count = _pick(feature, META, 9, tl.int32)
        truth_u = tl.load(Target + _offset(n, u, SHAPE, ts), n < N, other=0).to(tl.float32)
        valid = (n < N) & (tl.abs(truth_u) < float("inf"))
        if HAS_MASK:
            valid &= tl.load(Mask + _offset(n, u, SHAPE, ms), n < N, other=0) != 0
        reference = truth_u
        truth_v = tl.full((BLOCK,), 0.0, tl.float32)
        if v >= 0:
            truth_v = tl.load(Target + _offset(n, v, SHAPE, ts), n < N, other=0).to(tl.float32)
            valid &= tl.abs(truth_v) < float("inf")
            if HAS_MASK:
                valid &= tl.load(Mask + _offset(n, v, SHAPE, ms), n < N, other=0) != 0
            physical_u = truth_u * _pick(feature, META, 6) + _pick(feature, META, 5)
            physical_v = truth_v * _pick(feature, META, 8) + _pick(feature, META, 7)
            high = tl.maximum(tl.abs(physical_u), tl.abs(physical_v))
            ratio = tl.minimum(tl.abs(physical_u), tl.abs(physical_v)) / tl.where(high > 0, high, 1.0)
            reference = high * tl.sqrt(1.0 + ratio * ratio)
            valid &= tl.abs(reference) < float("inf")
        bins = tl.full((BLOCK,), 1.0, tl.float32)
        for i in tl.static_range(8):
            if i < cutoff_count:
                cutoff = _pick(feature, CUTOFFS, i)
                bins += tl.where(i < lower, reference < cutoff, reference > cutoff).to(tl.float32)
        value = _component(Output, Weight, Slope, n, u, slot_u, truth_u, valid, bins, N, SHAPE, os, ws, HAS_WEIGHT, SAVE_SLOPE, EPS)
        count = valid.to(tl.int32)
        if v >= 0:
            value += _component(Output, Weight, Slope, n, v, slot_v, truth_v, valid, bins, N, SHAPE, os, ws, HAS_WEIGHT, SAVE_SLOPE, EPS)
            count *= 2
        tile = feature.to(tl.int64) * tl.num_programs(0) + tl.program_id(0)
        tl.store(Partial + tile, tl.sum(value, 0))
        tl.store(Counts + tile, tl.sum(count, 0))

    @triton.jit
    def _gt_reduce(Partial: tl.tensor, Counts: tl.tensor, Loss: tl.tensor, InvCount: tl.tensor, PARTS: tl.constexpr, BLOCK: tl.constexpr) -> None:
        offsets = tl.arange(0, BLOCK)
        totals = tl.full((BLOCK,), 0.0, tl.float32)
        counts = tl.full((BLOCK,), 0, tl.int64)
        # Accumulate per lane first, then perform just one pair of block reductions.
        for start in range(tl.cdiv(PARTS, BLOCK)):
            idx = start * BLOCK + offsets
            totals += tl.load(Partial + idx, idx < PARTS, other=0)
            counts += tl.load(Counts + idx, idx < PARTS, other=0).to(tl.int64)
        inverse = 1.0 / tl.maximum(tl.sum(counts, 0), 1).to(tl.float32)
        tl.store(Loss, tl.sum(totals, 0) * inverse)
        tl.store(InvCount, inverse)

    @triton.jit
    def _gt_backward(
        Slope: tl.tensor,
        InvCount: tl.tensor,
        GradLoss: tl.tensor,
        GradOutput: tl.tensor,
        N: tl.constexpr,
        SHAPE: tl.constexpr,
        GS: tl.constexpr,
        CHANNELS: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        channel = tl.program_id(1)
        slot = tl.full((), -1, tl.int32)
        for i in tl.static_range(len(CHANNELS)):
            slot = tl.where(channel == CHANNELS[i], i, slot)
        gradient = tl.full((BLOCK,), 0.0, tl.float32)
        if slot >= 0:
            slope = tl.load(Slope + slot.to(tl.int64) * N + n, n < N, other=0).to(tl.float32)
            gradient = slope * (tl.load(InvCount) * tl.load(GradLoss).to(tl.float32))
        # Write every channel once, including zeros, instead of zeroing then scattering.
        tl.store(GradOutput + _offset(n, channel, SHAPE, GS), gradient, n < N)

    class _GTWeightedFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            output: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor | None,
            mask: torch.Tensor | None,
            specs: tuple[_FeatureSpec, ...],
            channels: tuple[int, ...],
            eps: float,
        ) -> torch.Tensor:
            has_weight, has_mask = weight is not None, mask is not None
            weight = weight if weight is not None else output
            mask = mask if mask is not None else output
            b, t, c, h, w = output.shape
            n = b * t * h * w
            tiles = triton.cdiv(n, 512)
            partial = torch.empty((tiles * len(specs),), device=output.device, dtype=torch.float32)
            counts = torch.empty_like(partial, dtype=torch.int32)
            loss = torch.empty((), device=output.device, dtype=torch.float32)
            inverse = torch.empty_like(loss)
            save_slope = ctx.needs_input_grad[0]
            # Store only selected-channel local derivatives, in the prediction dtype.
            slope = torch.empty((len(channels), n), device=output.device, dtype=output.dtype) if save_slope else None
            rows = [(channels[u], channels[v] if v >= 0 else -1, u, v, lower, mu, su, mv, sv, len(cutoffs)) for u, v, cutoffs, lower, mu, su, mv, sv in specs]
            metadata = tuple(zip(*rows))
            cutoffs = tuple(tuple(spec[2][i] if i < len(spec[2]) else 0.0 for spec in specs) for i in range(8))
            with torch.cuda.device(output.device):
                _gt_forward[(tiles, len(specs))](
                    output,
                    target,
                    weight,
                    mask,
                    partial,
                    counts,
                    slope,
                    N=n,
                    SHAPE=(b, t, h, w),
                    STRIDES=(output.stride(), target.stride(), weight.stride(), mask.stride()),
                    META=metadata,
                    CUTOFFS=cutoffs,
                    HAS_WEIGHT=has_weight,
                    HAS_MASK=has_mask,
                    SAVE_SLOPE=save_slope,
                    EPS=eps,
                    BLOCK=512,
                )
                _gt_reduce[(1,)](partial, counts, loss, inverse, PARTS=partial.numel(), BLOCK=1024)
            if slope is not None:
                ctx.save_for_backward(slope, inverse)
                ctx.output_shape = (b, t, c, h, w)
                ctx.channels = channels
            return loss

        @staticmethod
        @once_differentiable
        def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None, None]:
            slope, inverse = ctx.saved_tensors
            gradient = torch.empty(ctx.output_shape, device=slope.device, dtype=slope.dtype)
            b, t, c, h, w = ctx.output_shape
            n = b * t * h * w
            with torch.cuda.device(slope.device):
                _gt_backward[(triton.cdiv(n, 1024), c)](
                    slope, inverse, grad_loss, gradient, N=n, SHAPE=(b, t, h, w), GS=gradient.stride(), CHANNELS=ctx.channels, BLOCK=1024
                )
            return gradient, None, None, None, None, None, None

    return _GTWeightedFunction


class GTWeightedLoss(nn.Module):
    """GT-weighted Charbonnier mean for normalized (B, T, C, H, W).

    Weight = 1 + number of strictly crossed tail thresholds.
    Ascending GT: upper has weights 1,2,3,4,5; lower has 5,4,3,2,1;
    both has 5,4,3,2,1,2,3,4,5 (nine intervals, five weight levels).
    Equality belongs to the less heavily weighted side. Repeated thresholds
    can skip levels. Divide the weighted point-loss sum by valid element
    count, not by the sum of weights.

    Configured thresholds use denormalized training units (TP stays log1p).
    Upper/lower tails use four cutoffs; both tails use four per side.
    Wind components share weights based on physical speed. Other channels
    keep weight 1. Nonfinite GT and masked elements are excluded from the mean.
    Normalization constants and thresholds are ordinary Python values.
    CUDA FP16/BF16/FP32 uses fused Triton kernels with native storage and FP32
    register arithmetic/reduction, without full-grid FP32 temporaries. Scalar
    thresholds are pre-normalized. Requires CUDA and Triton; external weights
    are fixed. Fused rounding and normalized cutoffs can change boundary ties.
    Backward saves only selected-channel local derivatives in prediction dtype,
    introducing low-precision rounding before the final gradient scaling.
    """

    def __init__(
        self, channel_names: Sequence[str], mean: torch.Tensor, std: torch.Tensor, features: Mapping[str, Mapping[str, Any]], eps: float = 1e-6
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("GTWeightedLoss requires a CUDA GPU; CPU and MPS are not supported")
        self.channel_names = tuple(channel_names)
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be nonempty and unique")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        mean = mean.detach().reshape(-1)
        std = std.detach().reshape(-1)
        if mean.numel() != len(self.channel_names) or std.shape != mean.shape:
            raise ValueError("mean/std must contain one value per channel")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("mean/std must be finite and std positive")
        means: list[float] = mean.tolist()
        stds: list[float] = std.tolist()
        indices = {name: i for i, name in enumerate(self.channel_names)}
        specs: list[_FeatureSpec] = []
        assigned: set[int] = set()
        for name, specification in features.items():
            names = tuple(specification.get("channels", (name,)))
            if len(names) not in (1, 2) or len(set(names)) != len(names):
                raise ValueError("Each feature must specify one scalar channel or two wind components")
            present = [n in indices for n in names]
            if any(present) and not all(present):
                raise ValueError(f"Both wind components {names} are required")
            if not any(present):
                continue
            channels = tuple(indices[n] for n in names)
            if assigned.intersection(channels):
                raise ValueError(f"Channels assigned more than once: {names}")
            assigned.update(channels)
            mode = specification["mode"]
            thresholds = tuple(float(t) for t in specification["thresholds"])
            if mode not in {"upper", "lower", "both"}:
                raise ValueError("mode must be upper, lower or both")
            if len(thresholds) != (8 if mode == "both" else 4):
                raise ValueError(f"Wrong threshold count for {mode}")
            if not all(math.isfinite(t) for t in thresholds) or tuple(sorted(thresholds)) != thresholds:
                raise ValueError("Thresholds must be finite and nondecreasing")
            u = channels[0]
            v = channels[1] if len(channels) == 2 else -1
            lower_count = 4 if mode in {"lower", "both"} else 0
            # Scalars use normalized thresholds; wind pairs use physical speed.
            if v < 0:
                thresholds = tuple((value - means[u]) / stds[u] for value in thresholds)
            normalization = (0.0, 1.0, 0.0, 1.0) if v < 0 else (means[u], stds[u], means[v], stds[v])
            specs.append((u, v, thresholds, lower_count, *normalization))
        if not specs:
            raise ValueError("No threshold features match channel_names")
        unweighted: list[_FeatureSpec] = [(i, -1, (), 0, 0.0, 1.0, 0.0, 1.0) for i in range(len(self.channel_names)) if i not in assigned]
        self.kernel_specs = (*unweighted, *specs)
        self._backend = _load_triton_backend()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        channel_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Optionally read named channels directly from a full model output.

        channel_indices maps this loss's channel_names into output/target dim 2.
        External weight/mask broadcast against the full output before selection.
        """
        if output.shape != target.shape or output.ndim != 5:
            raise ValueError("output and target must have the same (B, T, C, H, W) shape")
        if not output.is_cuda or output.device != target.device:
            raise ValueError("output and target must be on the same CUDA device")
        if output.dtype not in (torch.float16, torch.bfloat16, torch.float32) or target.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("output and target must use FP16, BF16 or FP32")
        if weight is not None and weight.requires_grad:
            raise ValueError("GTWeightedLoss requires fixed external weights")
        if channel_indices is None:
            if output.shape[2] != len(self.channel_names):
                raise ValueError("output channels must match channel_names")
            channel_indices = tuple(range(len(self.channel_names)))
        if len(channel_indices) != len(self.channel_names) or len(set(channel_indices)) != len(channel_indices):
            raise ValueError("channel_indices must contain one unique index per named channel")
        if any(i < 0 or i >= output.shape[2] for i in channel_indices):
            raise ValueError("channel_indices is outside the model output")
        for value in (weight, mask):
            if value is not None and value.device != output.device:
                raise ValueError("weight/mask must be on the output device")
        # expand validates broadcasting without allocating full-grid copies.
        weight = weight.expand(output.shape) if weight is not None else None
        mask = mask.expand(output.shape) if mask is not None else None
        if output.numel() == 0:
            return output.sum(dtype=torch.float32) * 0
        return self._backend.apply(output, target.detach(), weight, mask, self.kernel_specs, channel_indices, self.eps)
