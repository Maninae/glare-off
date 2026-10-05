"""NAFNet building blocks rewritten from plain tensor ops so they export to browser-safe ONNX.

Follows NAFNet (Chen et al. 2022, MIT, github.com/megvii-research/NAFNet) with two export-driven
changes:

- `ChannelLayerNorm2d` normalizes over channels with mean / subtract / multiply / sqrt instead of
  NAFNet's custom `torch.autograd.Function`, which the ONNX exporter cannot trace. It exports to
  ReduceMean, Sub, Mul, Add, Sqrt, Div: all in onnxruntime-web's WebGPU table.
- The variance uses `centered * centered`, not `pow(2)`, so no Pow node is emitted.

Every other op (Conv, Split from `chunk`, GlobalAveragePool from `AdaptiveAvgPool2d(1)`, Mul,
Add) maps to a WebGPU-supported ONNX op; `export/webgpu_operator_inventory.py` checks this on
every export rather than trusting this docstring.
"""

import torch
from torch import nn

LAYER_NORM_EPSILON = 1e-6
# NAFNet's feed-forward expansion and depthwise expansion both double the channel count.
DEPTHWISE_EXPANSION_RATIO = 2
FEED_FORWARD_EXPANSION_RATIO = 2


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor, with per-channel affine.

    Equivalent to NAFNet's LayerNorm2d forward pass. Each pixel's channel vector is normalized to
    zero mean and unit variance independently, so the layer is resolution-independent and
    batch-independent (no BatchNorm statistics to drift between train and export).
    """

    def __init__(self, channel_count: int, epsilon: float = LAYER_NORM_EPSILON):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channel_count, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel_count, 1, 1))
        self.epsilon = epsilon

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Normalize each pixel's channel vector, then scale and shift per channel."""
        channel_mean = feature_map.mean(dim=1, keepdim=True)
        centered = feature_map - channel_mean
        channel_variance = (centered * centered).mean(dim=1, keepdim=True)
        normalized = centered / torch.sqrt(channel_variance + self.epsilon)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    """NAFNet's activation: split channels in half and multiply the halves (exports as Split + Mul)."""

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Return first_half * second_half along the channel axis."""
        first_half, second_half = feature_map.chunk(2, dim=1)
        return first_half * second_half


class SimplifiedChannelAttention(nn.Module):
    """NAFNet's SCA: global average pool, 1x1 conv, multiply back (no nonlinearity)."""

    def __init__(self, channel_count: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mixing_conv = nn.Conv2d(channel_count, channel_count, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Reweight channels by a linear function of their global means."""
        return feature_map * self.channel_mixing_conv(self.global_pool(feature_map))


class NAFBlock(nn.Module):
    """One NAFNet block: a spatial-mixing branch and a channel-mixing branch, each residual.

    Spatial branch: LN -> 1x1 expand -> 3x3 depthwise -> SimpleGate -> SCA -> 1x1 -> scaled add.
    Channel branch: LN -> 1x1 expand -> SimpleGate -> 1x1 -> scaled add.

    - `residual_scale_*` start at zero (as in NAFNet), so a fresh block is the identity map.
    """

    def __init__(self, channel_count: int):
        super().__init__()
        depthwise_channels = channel_count * DEPTHWISE_EXPANSION_RATIO
        feed_forward_channels = channel_count * FEED_FORWARD_EXPANSION_RATIO
        self.spatial_norm = ChannelLayerNorm2d(channel_count)
        self.spatial_expand_conv = nn.Conv2d(channel_count, depthwise_channels, kernel_size=1)
        self.spatial_depthwise_conv = nn.Conv2d(
            depthwise_channels, depthwise_channels, kernel_size=3, padding=1, groups=depthwise_channels
        )
        self.spatial_gate = SimpleGate()
        self.channel_attention = SimplifiedChannelAttention(depthwise_channels // 2)
        self.spatial_project_conv = nn.Conv2d(depthwise_channels // 2, channel_count, kernel_size=1)

        self.channel_norm = ChannelLayerNorm2d(channel_count)
        self.channel_expand_conv = nn.Conv2d(channel_count, feed_forward_channels, kernel_size=1)
        self.channel_gate = SimpleGate()
        self.channel_project_conv = nn.Conv2d(feed_forward_channels // 2, channel_count, kernel_size=1)

        self.residual_scale_spatial = nn.Parameter(torch.zeros(1, channel_count, 1, 1))
        self.residual_scale_channel = nn.Parameter(torch.zeros(1, channel_count, 1, 1))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply the spatial branch then the channel branch, each as a scaled residual."""
        spatial = self.spatial_norm(feature_map)
        spatial = self.spatial_depthwise_conv(self.spatial_expand_conv(spatial))
        spatial = self.channel_attention(self.spatial_gate(spatial))
        after_spatial = feature_map + self.spatial_project_conv(spatial) * self.residual_scale_spatial

        channel = self.channel_gate(self.channel_expand_conv(self.channel_norm(after_spatial)))
        return after_spatial + self.channel_project_conv(channel) * self.residual_scale_channel


def build_nafblock_stage(channel_count: int, block_count: int) -> nn.Sequential:
    """Return `block_count` NAFBlocks of width `channel_count` in sequence (empty is allowed)."""
    return nn.Sequential(*[NAFBlock(channel_count) for _ in range(block_count)])
