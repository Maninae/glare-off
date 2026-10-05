"""Hand-set head weights that make an UNTRAINED GlareRemovalNAFNet produce visibly sane output.

Used only to give the web app a real model file (true graph, true I/O, true latency) before any
training has happened. The heads read the raw input crop as their last 3 channels, so setting
the center tap on those channels and zeroing every feature weight gives closed-form outputs:

    clean_crop = clip(input * (1 - STUB_DARKEN_FRACTION), 0, 1)
    masks[0]   = sigmoid(GLARE_SLOPE * (luminance - GLARE_LUMINANCE_THRESHOLD))      soft bright-pixel detector
    masks[1]   = sigmoid(LOST_DETAIL_SLOPE * (luminance - LOST_DETAIL_LUMINANCE_THRESHOLD))
"""

import torch

from glare_model.architecture.glare_removal_nafnet import IMAGE_CHANNEL_COUNT, GlareRemovalNAFNet

STUB_DARKEN_FRACTION = 0.08
# Rec. 601 luma weights on sRGB values: a cheap brightness proxy, fine for a stub.
LUMA_WEIGHTS_RGB = (0.299, 0.587, 0.114)
GLARE_LUMINANCE_THRESHOLD = 0.78
GLARE_SLOPE = 25.0
LOST_DETAIL_LUMINANCE_THRESHOLD = 0.95
LOST_DETAIL_SLOPE = 60.0


def apply_stub_head_weights(model: GlareRemovalNAFNet) -> GlareRemovalNAFNet:
    """In place: overwrite both heads so outputs follow the module docstring formulas; return model."""
    restoration_weight = model.restoration_head.weight
    mask_weight = model.mask_head.weight
    first_image_channel = restoration_weight.shape[1] - IMAGE_CHANNEL_COUNT
    center = restoration_weight.shape[-1] // 2
    with torch.no_grad():
        restoration_weight.zero_()
        model.restoration_head.bias.zero_()
        for color_channel in range(IMAGE_CHANNEL_COUNT):
            restoration_weight[color_channel, first_image_channel + color_channel, center, center] = -STUB_DARKEN_FRACTION

        mask_weight.zero_()
        mask_slopes_and_thresholds = (
            (GLARE_SLOPE, GLARE_LUMINANCE_THRESHOLD),
            (LOST_DETAIL_SLOPE, LOST_DETAIL_LUMINANCE_THRESHOLD),
        )
        for mask_channel, (slope, threshold) in enumerate(mask_slopes_and_thresholds):
            for color_channel, luma_weight in enumerate(LUMA_WEIGHTS_RGB):
                mask_weight[mask_channel, first_image_channel + color_channel, center, center] = slope * luma_weight
            model.mask_head.bias[mask_channel] = -slope * threshold
    return model
