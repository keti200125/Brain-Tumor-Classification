"""SSPANet attention module for feature refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSPANet(nn.Module):
    """Hybrid channel/spatial/style attention with residual feature fusion."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be greater than zero")

        hidden_channels = max(channels // reduction, 1)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.horizontal_context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.vertical_context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.style_projection = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        channel_attention = self.channel_attention(torch.cat([avg_pool, max_pool], dim=1))
        x_channel = x * channel_attention

        height, width = x_channel.shape[-2:]
        horizontal_context = x_channel.mean(dim=2, keepdim=True)
        vertical_context = x_channel.mean(dim=3, keepdim=True)

        horizontal_features = self.horizontal_context(horizontal_context)
        vertical_features = self.vertical_context(vertical_context)
        horizontal_features = F.interpolate(
            horizontal_features,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        vertical_features = F.interpolate(
            vertical_features,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        style_mean = x_channel.mean(dim=1, keepdim=True)
        style_std = x_channel.std(dim=1, keepdim=True, unbiased=False)
        style_features = self.style_projection(torch.cat([style_mean, style_std], dim=1))

        fused_spatial_features = torch.cat(
            [horizontal_features, vertical_features, style_features],
            dim=1,
        )
        spatial_attention = self.spatial_fusion(fused_spatial_features)
        x_spatial = x_channel * spatial_attention
        return x + x_spatial
