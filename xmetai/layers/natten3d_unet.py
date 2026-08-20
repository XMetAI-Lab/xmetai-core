# Author: zhoushuang — 框架本体原创(底层 PyTorch / Hourglass-NATTEN 为开源组件)
"""可插 backbone:条件视频扩散的 UNet 骨架。

定义 ``BackBone`` **公共协议** —— 一切 backbone 都是 ``nn.Module``,约定::

    forward(tokens, position_info, context) -> tokens

    tokens:        ``(B, T, H, W, C)``  5D channels-last token 序列
    position_info: ``PositionInfo`` 对象(modality_id ``(B,T)`` long + time_coord ``(B,T)`` float)
                   —— 每个 slot 的语义坐标;backbone 自己决定怎么编码它
    context:       ``(B, C_ctx, H, W)``  空间条件图;backbone 自己决定怎么用

返回同形状 5D 张量。**公共协议不绑定任何 backbone 的内部机制** —— modality embedding、
时间编码(RoPE / sinusoidal / ...)、context 调制(AdaLN / FiLM / concat)全是子类自己的事。
这样 NATTEN / mamba / CNN 各自实现,公共接口不动。

当前实现:
- :class:`Natten3DBackbone` —— 入口 modality embedding + Hourglass-NATTEN 3D attention
  (per-sample 时间 RoPE,吃 ``time_coord``)+ Spatial AdaLN(吃 ``context``)+ PatchMerge/Split。
"""

from __future__ import annotations

from typing import Any, Dict, List

from torch import Tensor, nn

from xmetai.layers.embed import PatchMerge2D, PatchSplit2D
from xmetai.layers.natten import TransformerLayer3D

try:
    import natten  # noqa: F401  # Hourglass-NATTEN
except ImportError:
    natten = None

__all__ = ["BackBone", "Natten3DBackbone"]


class BackBone(nn.Module):
    """可插 backbone 公共协议(抽象概念)。

    ``forward(tokens, position_info, context) -> tokens``:
        tokens:        ``(B, T, H, W, C)`` token 序列。
        position_info: 有 ``.modality_id (B,T) long`` 与 ``.time_coord (B,T) float`` 的对象
                       (见 :class:`~xmetai.diffusion.layout.PositionInfo`)。
        context:       ``(B, C_ctx, H, W)`` 空间条件图。
    返回同形状 5D。具体怎么编码 modality / time、怎么用 context,由子类自定。
    """

    def forward(self, tokens: Tensor, position_info: Any, context: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class Natten3DBackbone(BackBone):
    """NATTEN 3D UNet 骨架。

    - 入口加 **modality embedding**(``position_info.modality_id`` → 加到 level0 token);
    - 每层 3D attention 用 **per-sample 时间 RoPE**(``position_info.time_coord``);
    - **Spatial AdaLN** 吃 ``context`` 空间图;
    - 按 ``levels`` 下采样 → 瓶颈 → 上采样(skip 连接)。
    """

    def __init__(
        self,
        levels: List[Dict[str, Any]],
        cond_dim: int,
        num_modalities: int,
        dim: int,
        num_states: int = 2,
        cosine_sim_scale: bool = True,
    ) -> None:
        super().__init__()
        if natten is None:
            raise ModuleNotFoundError("Hourglass-NATTEN is required. `pip install Hourglass-NATTEN`")

        self.cosine_sim_scale = cosine_sim_scale
        self.dim = int(dim)
        # modality emb + state emb(observed/missing/...;入口加到 level0 token)
        self.modality_embedding = nn.Embedding(int(num_modalities), self.dim)
        self.state_embedding = nn.Embedding(int(num_states), self.dim)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        num_levels = len(levels)
        for i, spec in enumerate(levels):
            block = nn.ModuleList([self._make_layer(spec, cond_dim) for _ in range(int(spec["depth"]))])
            if i < num_levels - 1:
                self.down_blocks.append(block)
                self.downsamplers.append(PatchMerge2D(int(spec["width"]), int(levels[i + 1]["width"])))
            else:
                self.mid_block = block
        for i in range(num_levels - 2, -1, -1):
            spec = levels[i]
            below_w = int(levels[i + 1]["width"])
            width = int(spec["width"])
            self.upsamplers.append(PatchSplit2D(below_w, width))
            self.up_blocks.append(nn.ModuleList([self._make_layer(spec, cond_dim) for _ in range(int(spec["depth"]))]))

    def _make_layer(self, spec: Dict[str, Any], cond_dim: int) -> TransformerLayer3D:
        return TransformerLayer3D(
            dim=int(spec["width"]),
            ff_dim=int(spec["d_ff"]),
            num_heads=int(spec["num_heads"]),
            cond_dim=cond_dim,
            kernel_size_thw=(
                int(spec["kernel_size_thw"][0]),
                int(spec["kernel_size_thw"][1]),
                int(spec["kernel_size_thw"][2]),
            ),
            dropout=float(spec.get("dropout", 0.0)),
            theta=int(spec.get("theta", 10000)),
            cosine_sim_scale=self.cosine_sim_scale,
            use_checkpoint=bool(spec.get("use_checkpoint", False)),
            use_global_attn=bool(spec.get("use_global_attn", False)),
            global_attn_heads=int(spec["global_attn_heads"]) if spec.get("global_attn_heads") is not None else None,
        )

    def _run_block(self, block: nn.ModuleList, x: Tensor, context: Tensor, time_coord: Tensor) -> Tensor:
        for layer in block:
            x = layer(x, context, time_coord)
        return x

    def forward(self, tokens: Tensor, position_info: Any, context: Tensor) -> Tensor:
        # 入口:modality emb + state emb(observed/missing)加到 level0 token
        mod_emb = self.modality_embedding(position_info.modality_id)  # (B, seq, dim)
        state_emb = self.state_embedding(position_info.state_id)  # (B, seq, dim)
        tokens = tokens + (mod_emb + state_emb)[:, :, None, None, :]  # broadcast (B, seq, H, W, dim)
        time_coord = position_info.time_coord  # (B, seq)

        skips: list = []
        for block, down in zip(self.down_blocks, self.downsamplers):
            tokens = self._run_block(block, tokens, context, time_coord)
            skips.append(tokens)
            tokens = down(tokens)
        tokens = self._run_block(self.mid_block, tokens, context, time_coord)
        for up, block in zip(self.upsamplers, self.up_blocks):
            skip = skips.pop()
            tokens = up(tokens, skip)
            tokens = self._run_block(block, tokens, context, time_coord)
        return tokens
