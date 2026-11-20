"""UNet3D with spatial downsampling and full-resolution skip fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """Compact residual UNet using spatial-only down/up-sampling."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 128):
        super().__init__()
        self.encoder1 = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.encoder2 = nn.Conv3d(base_channels, base_channels * 2, 3, padding=1)
        self.encoder3 = nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1)

        self.decoder1 = nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.decoder2 = nn.Conv3d(base_channels * 2, base_channels, 3, padding=1)
        self.fullres_decoder = nn.Conv3d(base_channels * 2, base_channels, 3, padding=1)
        self.head = nn.Conv3d(base_channels, out_channels, 1)
        self._init_weights()
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    @staticmethod
    def _upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        fullres_skip = F.relu(self.encoder1(x))
        skip1 = self._pool(fullres_skip)
        skip2 = F.relu(self._pool(self.encoder2(skip1)))
        x = F.relu(self._pool(self.encoder3(skip2)))

        x = F.relu(self._upsample(self.decoder1(x), skip2)) + skip2
        x = F.relu(self._upsample(self.decoder2(x), skip1)) + skip1
        x = self._upsample(x, fullres_skip)
        x = F.relu(self.fullres_decoder(torch.cat((x, fullres_skip), dim=1)))
        x = self.head(x)
        return x.permute(0, 2, 1, 3, 4).contiguous()
