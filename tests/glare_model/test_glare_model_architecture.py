"""Forward shapes, parameter counts per variant, and the identity-at-init property."""

import pytest
import torch

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet, count_trainable_parameters
from glare_model.architecture.stub_weights import STUB_DARKEN_FRACTION, apply_stub_head_weights

# Measured counts; the research doc (03, Q5) independently measured 1.14M and 2.93M for these shapes.
EXPECTED_PARAMETER_COUNTS = {
    "small": (dict(encoder_block_counts=[1, 1, 1, 1], middle_block_count=1), 1_137_628),
    "base": (dict(), 2_926_748),
    "large": (dict(base_width=24, middle_block_count=6), 8_625_188),
}


@pytest.mark.parametrize("height, width", [(64, 128), (256, 512), (96, 352), (16, 16)])
def test_forward_shapes_and_ranges_at_several_sizes(tiny_glare_model, height, width):
    glare_crop = torch.rand(2, 3, height, width)
    with torch.no_grad():
        clean_crop, masks = tiny_glare_model(glare_crop)
    assert clean_crop.shape == (2, 3, height, width)
    assert masks.shape == (2, 2, height, width)
    assert clean_crop.min() >= 0 and clean_crop.max() <= 1
    assert masks.min() >= 0 and masks.max() <= 1


@pytest.mark.parametrize("variant_name", sorted(EXPECTED_PARAMETER_COUNTS))
def test_parameter_counts_per_variant(variant_name):
    constructor_kwargs, expected_count = EXPECTED_PARAMETER_COUNTS[variant_name]
    assert count_trainable_parameters(GlareRemovalNAFNet(**constructor_kwargs)) == expected_count


def test_fresh_model_is_exact_identity_on_clean_crop(tiny_model_kwargs):
    model = GlareRemovalNAFNet(**tiny_model_kwargs).eval()
    glare_crop = torch.rand(1, 3, 64, 128)
    with torch.no_grad():
        clean_crop, masks = model(glare_crop)
    assert torch.equal(clean_crop, glare_crop)
    assert masks.max() < 0.05  # mask head starts near sigmoid(-4)


def test_stub_weights_darken_and_detect_bright_pixels(tiny_model_kwargs):
    model = apply_stub_head_weights(GlareRemovalNAFNet(**tiny_model_kwargs)).eval()
    glare_crop = torch.full((1, 3, 32, 64), 0.3)
    glare_crop[..., :16, :32] = 1.0
    with torch.no_grad():
        clean_crop, masks = model(glare_crop)
    torch.testing.assert_close(clean_crop[..., 20:, 40:], glare_crop[..., 20:, 40:] * (1 - STUB_DARKEN_FRACTION))
    assert masks[0, 0, 4, 4] > 0.95 and masks[0, 0, 28, 60] < 0.01
