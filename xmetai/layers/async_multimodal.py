# Author: zhoushuang — 框架本体原创(底层 PyTorch / Hourglass-NATTEN 为开源组件)
"""多模态异步:声明 slot 序列 → 编码 → 组装成时空序列 → backbone 融合。

- :class:`SlotSpec` / :class:`Layout`:声明 slot 序列 ``(modality, time)``,产出
  :class:`PositionInfo`(modality_id + time_coord + state_id)。
- :class:`FeatureEncoder`:``frames + availability → tokens``(模态特征编码)。
- :class:`AsyncMultiModal`:``tokens + availability + time_coords + context → 全序列 token``(组装 + backbone 融合)。
- :class:`AsyncMultiModalModel`:封装 encoders + AsyncMultiModal;L3 给 ``frames + availability +
  time_coords + context``,内部切片位置 + encode + 组装(调用方不碰位置索引)。

对接::

    model = AsyncMultiModalModel(layout, encoders, backbone, dim)
    out = model(frames_by_modality, availability_flags, time_coords, context)   # (B, seq, H2, W2, dim)

核心只提供:``modality_id`` / ``state_id``(backbone 输入 embedding)、``time_coord``(temporal RoPE)、
``tokens`` / ``availability_flags`` / ``context``。AdaLN / DOY / HOD / 扩散 timestep / 地形静态场
属于具体 backbone / 任务模型,经 ``context`` 注入,不写死在框架。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from xmetai.layers.embed import TemporalPatchEmbed

if TYPE_CHECKING:
    from xmetai.layers.natten3d_unet import BackBone

__all__ = [
    "NULL_MODALITY",
    "STATE_OBSERVED",
    "STATE_MISSING",
    "SlotSpec",
    "Availability",
    "PositionInfo",
    "Layout",
    "HEBEI_7FRAME",
    "FeatureEncoder",
    "AsyncMultiModal",
    "AsyncMultiModalModel",
]

NULL_MODALITY = "<null>"
STATE_OBSERVED = 0
STATE_MISSING = 1


# ======================================================================================
# slot 序列声明 + 语义坐标
# ======================================================================================
@dataclass(frozen=True)
class SlotSpec:
    """序列中一个 token 槽位:(modality, time)。

    Attributes:
        modality: 模态名,如 ``"noise"`` / ``"coarse"``。
        time: 逻辑时间 offset(build 时排序;真实物理时间由 dataset 算成 ``(B, seq)`` 传入)。
    """

    modality: str
    time: int

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or not self.modality:
            raise ValueError("modality 必须是非空字符串")
        if self.modality == NULL_MODALITY:
            raise ValueError(f"modality 不能用保留名 {NULL_MODALITY!r}")
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            raise TypeError("time 必须是 int")


@dataclass
class Availability:
    """per-sample 每 slot 数据可用性 ``(B, seq)`` bool(helper:核心/position_info 接 ``flags: Tensor``)。

    ``False`` → 缺数据(state=missing)。可由 dataset 的 ``batch["availability"]`` 直接构造。
    """

    flags: Tensor

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.flags):
            raise TypeError("Availability.flags 必须是 torch.Tensor")
        if self.flags.dim() != 2:
            raise ValueError(f"flags 必须是 (B, seq) 2D,得到 {tuple(self.flags.shape)}")
        if self.flags.dtype != torch.bool:
            object.__setattr__(self, "flags", self.flags.to(torch.bool))

    @classmethod
    def all_available(cls, B: int, seq: int, device: Any = None) -> "Availability":
        return cls(torch.ones(B, seq, dtype=torch.bool, device=device))

    @classmethod
    def none_available(cls, B: int, seq: int, device: Any = None) -> "Availability":
        return cls(torch.zeros(B, seq, dtype=torch.bool, device=device))

    @property
    def batch_size(self) -> int:
        return int(self.flags.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.flags.shape[1])


def _to_slotspec(s: Union[SlotSpec, "Any"]) -> SlotSpec:
    if isinstance(s, SlotSpec):
        return s
    return SlotSpec(modality=s["modality"], time=s["time"])


class PositionInfo(NamedTuple):
    """每 slot 的语义坐标(给 backbone 编码)。

    Attributes:
        modality_id: ``(B, seq)`` long —— 原 modality id(**缺失也不变**,保留模态信息)。
        time_coord:  ``(B, seq)`` float —— per-sample 真实物理时间(任意实数;Layout 按 time 排序,
                      故沿 seq 非递减,NATTEN 的 temporal 邻居与真实时间一致)。
        state_id:    ``(B, seq)`` long —— observed/missing(从 availability 派生)。
    """

    modality_id: Tensor
    time_coord: Tensor
    state_id: Tensor


class Layout(nn.Module):
    """slot 序列拓扑 + 语义坐标产出(无参数)。

    ``__init__`` 内部按 ``time`` 稳定排序(同 time 按声明顺序)—— 保证 slot index 沿序列是时间
    非递减,NATTEN 的 temporal 邻居与真实时间一致。:meth:`position_info` 产出 :class:`PositionInfo`:
    **modality_id 保留原模态**(缺失不丢模态),``state_id`` 标 observed/missing。
    """

    def __init__(
        self,
        slots: Sequence[SlotSpec],
        modalities: Optional[Sequence[str]] = None,
        null_modality: str = NULL_MODALITY,
    ) -> None:
        super().__init__()
        self.null_modality = null_modality
        self.slots: list[SlotSpec] = sorted([_to_slotspec(s) for s in slots], key=lambda s: s.time)
        self.seq_len: int = len(self.slots)
        if self.seq_len == 0:
            raise ValueError("slots 不能为空")

        seen = {null_modality}
        if modalities is not None:
            for m in modalities:
                if m != null_modality:
                    seen.add(m)
        for s in self.slots:
            seen.add(s.modality)
        vocab = sorted(seen)
        self._modalities: list[str] = vocab
        self._modality_to_id: dict[str, int] = {m: i for i, m in enumerate(vocab)}
        self._null_id: int = self._modality_to_id[null_modality]
        slot_mod_ids = [self._modality_to_id[s.modality] for s in self.slots]
        slot_time_offsets = [float(s.time) for s in self.slots]
        # buffer(避免 forward 内动态建 tensor → compile 友好)
        self.register_buffer("slot_mod_ids_buffer", torch.tensor(slot_mod_ids, dtype=torch.long))
        self.register_buffer("slot_time_offsets_buffer", torch.tensor(slot_time_offsets, dtype=torch.float32))
        self._mod_positions: Dict[str, List[int]] = {}
        for i, s in enumerate(self.slots):
            self._mod_positions.setdefault(s.modality, []).append(i)

    @property
    def modalities(self) -> list[str]:
        return list(self._modalities)

    @property
    def num_modalities(self) -> int:
        return len(self._modalities)

    @property
    def null_id(self) -> int:
        return self._null_id

    def slot_mod_ids(self) -> Tensor:
        return self.slot_mod_ids_buffer

    def slot_indices(self, modality: str, time: Optional[int] = None) -> tuple[int, ...]:
        """查 (modality[, time]) 在序列中的位置(同一可重复,返回全部 index)。"""
        if time is None:
            return tuple(self._mod_positions.get(modality, ()))
        return tuple(i for i, s in enumerate(self.slots) if s.modality == modality and s.time == time)

    def position_info(self, time_coords: Tensor, availability_flags: Optional[Tensor] = None) -> PositionInfo:
        """``time_coords (B, seq)`` [+ availability_flags (B, seq) bool] → :class:`PositionInfo`。

        modality_id 保留原模态(缺失也不变);state_id = observed/missing(从 availability_flags)。
        """
        if time_coords.dim() != 2 or time_coords.shape[1] != self.seq_len:
            raise ValueError(f"time_coords 必须是 (B, {self.seq_len}),得到 {tuple(time_coords.shape)}")
        B = time_coords.shape[0]
        ids = self.slot_mod_ids_buffer.to(time_coords.device).unsqueeze(0).expand(B, self.seq_len)
        if availability_flags is None:
            state = torch.zeros(B, self.seq_len, dtype=torch.long, device=time_coords.device)
        else:
            if availability_flags.shape != (B, self.seq_len):
                raise ValueError(f"availability_flags 必须是 ({B}, {self.seq_len})")
            state = torch.where(availability_flags, STATE_OBSERVED, STATE_MISSING).long()
        return PositionInfo(modality_id=ids, time_coord=time_coords, state_id=state)

    def default_time_coords(self, B: int, device: Any = None) -> Tensor:
        """用 ``slot.time`` 当坐标(单位 1),``(B, seq)`` float。真实物理时间由 dataset 算。"""
        return self.slot_time_offsets_buffer.to(device).unsqueeze(0).expand(B, self.seq_len).contiguous()


def HEBEI_7FRAME() -> list[SlotSpec]:
    """河北 7 帧降尺度布局范例(10 slot:init@0, noise@0..6, coarse@3, coarse@6;声明顺序即同 time 次序)。"""
    return [
        SlotSpec("init", 0),
        SlotSpec("noise", 0),
        SlotSpec("noise", 1),
        SlotSpec("noise", 2),
        SlotSpec("coarse", 3),
        SlotSpec("noise", 3),
        SlotSpec("noise", 4),
        SlotSpec("noise", 5),
        SlotSpec("coarse", 6),
        SlotSpec("noise", 6),
    ]


# ======================================================================================
# 模态特征编码
# ======================================================================================
class FeatureEncoder(nn.Module):
    """模态特征编码:``frames (B, K, H, W, C) + availability (B, K)`` → ``tokens (B, K, H2, W2, dim)``。

    接纳:``frames`` 各模态 raw 场(5D channels-last,K=该模态 slot 数);``availability`` per-slot 有无。
    输出:patch-embed 后的 token ``(B, K, H2, W2, dim)``(H2=H//patch, W2=W//patch)。

    缺失处理(安全):先 ``where(avail, frames, 0)`` 把缺失 raw 置零(避免 NaN/垃圾进 patch),再 patch-embed,
    最后把缺失位置 token 替换成 ``missing_token``(learnable 或零)—— **缺失位置始终覆盖**(绝不漏 patch 垃圾)。
    patch-embed 用 ``TemporalPatchEmbed``(per-frame Conv2d)。

    例子::

        enc = FeatureEncoder("coarse", in_ch=67, dim=256, patch_size_hw=(2, 2))
        tok = enc(coarse_frames, avail)   # coarse_frames (B,2,H,W,67), avail (B,2) → tok (B,2,H2,W2,256)

        # 多源(如 init 冷/热):继承 override forward,L3 reader 把多源拼成 frames 传入
        class HebeiInitFeatureEncoder(FeatureEncoder):
            def forward(self, frames, availability, aux=None): ...   # frames 含 coarse_t0+sat+highres

    Args:
        modality: 模态名(供组装时映射到序列位置)。
        in_ch: 数据通道数。
        dim: token 维(= backbone token 维)。
        patch_size_hw: patch-embed 的 patch 大小。
        missing_token_learnable: 缺数据 slot 用 learnable missing_token(否则零)。
    """

    def __init__(
        self,
        modality: str,
        in_ch: int,
        dim: int,
        patch_size_hw: Union[List[int], Tuple[int, int]] = (2, 2),
        missing_token_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.dim = int(dim)
        # patch-embed(per-frame Conv2d):(B,K,H,W,C)→(B,K,H2,W2,dim)
        self.feature = TemporalPatchEmbed(in_ch, dim, patch_size=patch_size_hw)
        self.missing_token_learnable = bool(missing_token_learnable)
        if self.missing_token_learnable:
            self.missing_token = nn.Parameter(torch.zeros(self.dim))
            nn.init.normal_(self.missing_token, std=0.02)

    def forward(self, frames: Tensor, availability: Tensor, aux: Optional[Tensor] = None) -> Tensor:
        """frames (B, K, H, W, C) + availability (B, K) bool → tokens (B, K, H2, W2, dim)。"""
        if frames.shape[:2] != availability.shape:
            raise ValueError(f"frames (B,K,...)={tuple(frames.shape[:2])} 与 availability (B,K)={tuple(availability.shape)} 不符")
        mask = availability[:, :, None, None, None]  # (B, K, 1, 1, 1)
        frames_safe = torch.where(mask, frames, torch.zeros_like(frames))  # 缺失 raw 置零(安全)
        feat = self.feature(frames_safe)  # (B, K, H2, W2, dim)
        B, K = availability.shape
        H2, W2 = feat.shape[2], feat.shape[3]
        if self.missing_token_learnable:
            missing = self.missing_token.view(1, 1, 1, 1, self.dim).expand(B, K, H2, W2, self.dim)
        else:
            missing = torch.zeros_like(feat)
        return torch.where(mask, feat, missing)  # 缺失位置始终覆盖


# ======================================================================================
# 异步多模态组装核心
# ======================================================================================
class AsyncMultiModal(nn.Module):
    """多模态异步组装:``tokens + availability + time_coords + context → 全序列 token``。

    接纳:各模态**已提取**的 token ``{modality: (B, K_mod, H2, W2, dim)}``(由 FeatureEncoder 或 L3 任意
    encoder 产出);``availability_flags (B, seq) bool``、``time_coords (B, seq) float``、``context`` 可选 ``(B, C_ctx, H2, W2)``。
    输出:全序列 token ``(B, seq, H2, W2, dim)``(未 decode)。

    核心**不强绑 FeatureEncoder**:L3 可先各自提取(surface/atmos 双流等)再融合成 tokens,直接喂本核心。
    流程:scatter 各模态 token 到序列(layout 位置)→ :meth:`Layout.position_info` → ``backbone(tokens, position_info, context)``。

    例子::

        out = core(
            tokens_by_modality={"noise": noise_tok, "coarse": coarse_tok, "init": init_tok},
            availability_flags=avail,   # (B, seq)
            time_coords=tc,             # (B, seq)
            context=ctx,                # (B, C_ctx, H2, W2) 或 None
        )   # → (B, seq, H2, W2, dim)

    Args:
        layout: :class:`Layout`。
        backbone: :class:`BackBone`(NATTEN / Transformer / Mamba ...),构造时注入。
        dim: token 维(= backbone token 维)。
    """

    def __init__(self, layout: Layout, backbone: BackBone, dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.backbone = backbone
        self.dim = int(dim)

    @property
    def seq_len(self) -> int:
        return self.layout.seq_len

    def forward(
        self,
        tokens_by_modality: Dict[str, Tensor],
        availability_flags: Tensor,
        time_coords: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """组装 + 融合 → 全序列 token ``(B, seq, H2, W2, dim)``(未 decode)。"""
        seq = self.layout.seq_len
        dev = availability_flags.device
        if availability_flags.dim() != 2 or availability_flags.shape[1] != seq:
            raise ValueError(f"availability_flags 必须是 (B, {seq}),得到 {tuple(availability_flags.shape)}")
        if time_coords.shape != availability_flags.shape or time_coords.device != dev:
            raise ValueError("time_coords 与 availability_flags 的 shape/device 必须一致")
        missing = [m for m in self.layout._mod_positions if m not in tokens_by_modality]
        if missing:
            raise ValueError(f"tokens_by_modality 缺 modality: {missing}")

        sample = next(iter(tokens_by_modality.values()))  # (B, K, H2, W2, dim)
        B, _K, H2, W2, _C = sample.shape
        if _C != self.dim:
            raise ValueError(f"tokens dim={_C} 与核心 dim={self.dim} 不符")
        x = sample.new_zeros(B, seq, H2, W2, self.dim)
        for mod, positions in self.layout._mod_positions.items():
            tok = tokens_by_modality[mod]
            if tok.shape[1] != len(positions):
                raise ValueError(f"modality '{mod}' tokens K={tok.shape[1]} != layout 该 modality slot 数 {len(positions)}")
            # 校验各模态 token 同 B/H2/W2/dim/device/dtype
            if tok.shape[0] != B or tok.shape[2:] != (H2, W2, self.dim) or tok.device != dev:
                raise ValueError(f"modality '{mod}' tokens 的 B/H2/W2/dim/device 与其它不一致")
            x[:, positions] = tok

        position_info = self.layout.position_info(time_coords, availability_flags)
        return self.backbone(x, position_info, context)


# ======================================================================================
# 上层封装:L3 给 frames,不碰位置索引
# ======================================================================================
class AsyncMultiModalModel(nn.Module):
    """封装 FeatureEncoders + AsyncMultiModal:调用方只给 frames,内部切片 + encode + 组装。

    接纳:``frames_by_modality {modality: (B, K_mod, H, W, C)}``(L3 reader 取好的 raw 场);
    ``availability_flags (B, seq) bool``、``time_coords (B, seq) float``、``context`` 可选 ``(B, C_ctx, H2, W2)``。
    输出:全序列 token ``(B, seq, H2, W2, dim)``。

    适合「每模态 raw 场 → 内部统一 FeatureEncoder」的常见路径。若要「先各自提取再融合」(如
    surface/atmos 双流),二选一:① 继承 FeatureEncoder override forward(多源);② L3 自己提好
    tokens 直接用 :class:`AsyncMultiModal`(绕过本封装)。

    例子::

        out = model(
            frames_by_modality={"noise": noise_frames, "coarse": coarse_frames},  # 各 (B, K, H, W, C)
            availability_flags=avail, time_coords=tc, context=ctx,
        )   # → (B, seq, H2, W2, dim)

    Args:
        layout: :class:`Layout`。
        feature_encoders: list[:class:`FeatureEncoder`](按 ``.modality`` 映射;必须与 layout 模态一一对应)。
        backbone: :class:`BackBone`,注入。
        dim: token 维。
    """

    def __init__(
        self,
        layout: Layout,
        feature_encoders: Sequence[FeatureEncoder],
        backbone: BackBone,
        dim: int,
    ) -> None:
        super().__init__()
        self.core = AsyncMultiModal(layout, backbone, int(dim))
        enc_mods = [fe.modality for fe in feature_encoders]
        if len(set(enc_mods)) != len(enc_mods):
            raise ValueError(f"feature_encoders 有重复 modality: {enc_mods}")
        layout_mods = set(layout._mod_positions)
        if set(enc_mods) != layout_mods:
            extra = set(enc_mods) - layout_mods
            lacking = layout_mods - set(enc_mods)
            raise ValueError(f"feature_encoders 与 layout 模态不对应: encoder 多 {extra} / 缺 {lacking}")
        self.feature_encoders = nn.ModuleDict({fe.modality: fe for fe in feature_encoders})
        # 预缓存每 modality 的序列位置(避免 forward 内动态 torch.tensor → compile 友好)
        self._mod_positions_sorted = sorted(layout._mod_positions.items())
        for mod, positions in layout._mod_positions.items():
            self.register_buffer(f"_pos_{mod}", torch.tensor(positions, dtype=torch.long), persistent=False)

    @property
    def layout(self) -> Layout:
        return self.core.layout

    @property
    def seq_len(self) -> int:
        return self.core.seq_len

    def forward(
        self,
        frames_by_modality: Dict[str, Tensor],
        availability_flags: Tensor,
        time_coords: Tensor,
        context: Optional[Tensor] = None,
        aux_map: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        dev = availability_flags.device
        tokens: Dict[str, Tensor] = {}
        for mod, enc in self.feature_encoders.items():
            pos = getattr(self, f"_pos_{mod}").to(dev)  # 缓存 buffer,去动态 torch.tensor
            avail_mod = availability_flags.index_select(1, pos)  # (B, K_mod)
            tokens[mod] = enc(frames_by_modality[mod], avail_mod, aux_map.get(mod) if aux_map else None)
        return self.core(tokens, availability_flags, time_coords, context)
