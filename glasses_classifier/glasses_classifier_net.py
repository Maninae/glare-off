"""GlassesClassifierNet: a ~200k-parameter CNN that says whether a face in the aligned eye crop wears glasses.

Input is the same tensor the glare model receives (float32 [B, 3, 256, 512], sRGB in [0, 1]);
output is the glasses probability [B, 1] with the sigmoid inside the graph.

- A 4x4 average pool first brings the crop to 64x128. Frames are large, high-contrast structures,
  so the gate needs no fine detail, and every later layer runs on 1/16 of the pixels.
- Then a plain conv stem and MobileNet-style depthwise-separable blocks (BatchNorm + ReLU), down
  to 4x8, a global mean, and a linear head.
- Every op exports to Conv / BatchNormalization / Relu / AveragePool / ReduceMean / Gemm / Sigmoid,
  all in onnxruntime-web's WebGPU table (checked on every export).
"""

import torch
from torch import nn

INPUT_POOL_FACTOR = 4
# (output channels, stride) per depthwise-separable block after the stem.
DEFAULT_BLOCK_SPECS: tuple[tuple[int, int], ...] = ((32, 1), (48, 2), (64, 1), (96, 2), (128, 1), (160, 2), (192, 1))
DEFAULT_STEM_CHANNELS = 24


class ConvBatchNormRelu(nn.Sequential):
    """Conv2d (no bias) -> BatchNorm2d -> ReLU; `groups=in_channels` makes it depthwise."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """3x3 depthwise conv (carries the stride) then 1x1 pointwise conv; residual when shapes match."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = ConvBatchNormRelu(in_channels, in_channels, 3, stride, groups=in_channels)
        self.pointwise = ConvBatchNormRelu(in_channels, out_channels, 1)
        self.uses_residual = stride == 1 and in_channels == out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the block (with the identity shortcut when shapes allow)."""
        block_output = self.pointwise(self.depthwise(features))
        return features + block_output if self.uses_residual else block_output


class GlassesClassifierNet(nn.Module):
    """Eye crop -> probability that the face wears glasses (eyeglasses or sunglasses).

    Args:
        stem_channels: channels of the stride-2 3x3 stem conv.
        block_specs: (out_channels, stride) per depthwise-separable block.
        dropout_probability: dropout before the linear head (training only).
    """

    def __init__(
        self,
        stem_channels: int = DEFAULT_STEM_CHANNELS,
        block_specs: tuple[tuple[int, int], ...] = DEFAULT_BLOCK_SPECS,
        dropout_probability: float = 0.2,
    ):
        super().__init__()
        self.input_pool = nn.AvgPool2d(INPUT_POOL_FACTOR, INPUT_POOL_FACTOR)
        layers: list[nn.Module] = [ConvBatchNormRelu(3, stem_channels, 3, stride=2)]
        in_channels = stem_channels
        for out_channels, stride in block_specs:
            layers.append(DepthwiseSeparableBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier_head = nn.Linear(in_channels, 1)

    def forward_logits(self, eye_crop: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] crop -> [B, 1] pre-sigmoid logit (what the training loss uses)."""
        features = self.feature_extractor(self.input_pool(eye_crop))
        pooled_features = features.mean(dim=(2, 3))
        return self.classifier_head(self.dropout(pooled_features))

    def forward(self, eye_crop: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] crop -> [B, 1] glasses probability (the exported graph)."""
        return torch.sigmoid(self.forward_logits(eye_crop))


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters with requires_grad."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_glasses_classifier_from_config(model_config: dict) -> GlassesClassifierNet:
    """Build the net from the `MODEL` config node (STEM_CHANNELS, BLOCK_SPECS, DROPOUT)."""
    return GlassesClassifierNet(
        stem_channels=int(model_config["STEM_CHANNELS"]),
        block_specs=tuple((int(channels), int(stride)) for channels, stride in model_config["BLOCK_SPECS"]),
        dropout_probability=float(model_config["DROPOUT"]),
    )
