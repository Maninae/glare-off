"""GlareRemovalNAFNet: the eye-crop network that predicts a clean crop plus glare and lost-detail masks.

Shape: a 4-level NAFNet U-Net (so H and W must be multiples of 16), fully convolutional, with two
3x3 heads that read the last decoder features concatenated with the raw input crop.

Outputs, per the Model I/O contract in the repo CLAUDE.md:
    clean_crop = clip(glare_crop + restoration_delta, 0, 1)      [N, 3, H, W]
    masks      = sigmoid(mask_logits)                              [N, 2, H, W] (glare, lost-detail)

Design decisions (reasons in glare_model/CLAUDE.md):
- Residual, not direct: the restoration head predicts a delta and is zero-initialized, so an
  untrained network is exactly the identity and the easy majority of pixels (no glare) costs
  nothing to learn.
- The mask GATE is not in the graph: the caller blends `glare_crop + glare_mask * (clean - glare)`.
  Training applies the same blend before the main loss (`losses/glare_removal_loss.py`).
- Heads see the raw input so brightness and saturation are directly available to the mask head,
  and so `stub_weights.py` can build a meaningful untrained stub by setting only head weights.
"""

import torch
from torch import nn

from glare_model.architecture.nafnet_blocks import build_nafblock_stage
from glare_model.registry import MODEL_REGISTRY

IMAGE_CHANNEL_COUNT = 3
MASK_CHANNEL_COUNT = 2
# Glare is rare per pixel, so start the mask head predicting "no glare" (sigmoid(-4) ~ 0.018),
# the focal-loss prior trick from RetinaNet.
DEFAULT_MASK_LOGIT_BIAS = -4.0
HEAD_KERNEL_SIZE = 3


@MODEL_REGISTRY.register
class GlareRemovalNAFNet(nn.Module):
    """NAFNet U-Net with a residual restoration head and a two-channel mask head.

    Args:
        base_width: channel count at full resolution; doubles at each of the 4 downsamplings.
        encoder_block_counts: NAFBlocks per encoder level (4 entries, full resolution first).
        middle_block_count: NAFBlocks at 1/16 resolution.
        decoder_block_counts: NAFBlocks per decoder level (4 entries, coarsest first).
        mask_logit_bias: initial bias of the mask head.

    - Required input multiple: `self.required_size_multiple` (16).
    """

    def __init__(
        self,
        base_width: int = 16,
        encoder_block_counts: list[int] | tuple[int, ...] = (1, 1, 2, 4),
        middle_block_count: int = 4,
        decoder_block_counts: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        mask_logit_bias: float = DEFAULT_MASK_LOGIT_BIAS,
    ):
        super().__init__()
        if len(encoder_block_counts) != len(decoder_block_counts):
            raise ValueError("encoder and decoder need the same number of levels")
        level_count = len(encoder_block_counts)
        self.required_size_multiple = 2**level_count

        self.intro_conv = nn.Conv2d(IMAGE_CHANNEL_COUNT, base_width, kernel_size=3, padding=1)
        self.encoder_stages = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        level_width = base_width
        for block_count in encoder_block_counts:
            self.encoder_stages.append(build_nafblock_stage(level_width, block_count))
            self.downsample_convs.append(nn.Conv2d(level_width, level_width * 2, kernel_size=2, stride=2))
            level_width *= 2

        self.middle_stage = build_nafblock_stage(level_width, middle_block_count)

        # Upsample = 1x1 conv to 2C channels, then PixelShuffle(2) to C/2 channels (DepthToSpace).
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for block_count in decoder_block_counts:
            self.upsample_layers.append(
                nn.Sequential(nn.Conv2d(level_width, level_width * 2, kernel_size=1, bias=False), nn.PixelShuffle(2))
            )
            level_width //= 2
            self.decoder_stages.append(build_nafblock_stage(level_width, block_count))

        head_input_channels = base_width + IMAGE_CHANNEL_COUNT
        self.restoration_head = nn.Conv2d(
            head_input_channels, IMAGE_CHANNEL_COUNT, kernel_size=HEAD_KERNEL_SIZE, padding=HEAD_KERNEL_SIZE // 2
        )
        self.mask_head = nn.Conv2d(
            head_input_channels, MASK_CHANNEL_COUNT, kernel_size=HEAD_KERNEL_SIZE, padding=HEAD_KERNEL_SIZE // 2
        )
        nn.init.zeros_(self.restoration_head.weight)
        nn.init.zeros_(self.restoration_head.bias)
        nn.init.normal_(self.mask_head.weight, std=1e-3)
        nn.init.constant_(self.mask_head.bias, mask_logit_bias)

    def predict_delta_and_mask_logits(self, glare_crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the U-Net and both heads; return (restoration_delta, mask_logits), both unbounded.

        Training uses these raw outputs (logits for a numerically stable focal loss, an unclipped
        delta so clipping never zeroes a gradient).
        """
        features = self.intro_conv(glare_crop)
        skip_features = []
        for encoder_stage, downsample_conv in zip(self.encoder_stages, self.downsample_convs):
            features = encoder_stage(features)
            skip_features.append(features)
            features = downsample_conv(features)

        features = self.middle_stage(features)

        for upsample_layer, decoder_stage, skip in zip(
            self.upsample_layers, self.decoder_stages, reversed(skip_features)
        ):
            features = decoder_stage(upsample_layer(features) + skip)

        head_input = torch.cat([features, glare_crop], dim=1)
        return self.restoration_head(head_input), self.mask_head(head_input)

    def forward(self, glare_crop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (clean_crop, masks) exactly as the exported ONNX graph does."""
        restoration_delta, mask_logits = self.predict_delta_and_mask_logits(glare_crop)
        clean_crop = torch.clamp(glare_crop + restoration_delta, 0.0, 1.0)
        return clean_crop, torch.sigmoid(mask_logits)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in `model`."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
