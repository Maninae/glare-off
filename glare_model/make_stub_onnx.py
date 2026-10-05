"""Entry point: `python -m glare_model.make_stub_onnx [--model-variant base] [--out app/models/glare_removal.onnx]`.

Writes an UNTRAINED stub with the real architecture and the real I/O signature, so the web app
can be built and timed before training finishes. Outputs follow closed-form rules
(`architecture/stub_weights.py`): `clean_crop` is the input darkened by 8%, mask channel 0 is a
soft bright-pixel detector, channel 1 a stricter near-white detector. Replace it with
`python -m glare_model.export_onnx` once a trained checkpoint exists.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from glare_model.architecture.stub_weights import apply_stub_head_weights
from glare_model.export.export_and_verify import export_and_verify_glare_model
from glare_model.training.component_builders import build_glare_model

logger = logging.getLogger(__name__)

MODEL_VARIANT_DIRECTORY = Path("configs/glare_model/model")
STUB_SEED = 0


def parse_command_line() -> argparse.Namespace:
    """Parse the model variant and output path."""
    parser = argparse.ArgumentParser(description="Write an untrained-but-sane stub ONNX model.")
    parser.add_argument("--model-variant", default="base", help="small | base | large")
    parser.add_argument("--out", type=Path, default=Path("app/models/glare_removal.onnx"))
    return parser.parse_args()


def main() -> None:
    """Build the variant, set stub head weights, export, verify, print the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line()
    torch.manual_seed(STUB_SEED)
    variant_config = OmegaConf.load(MODEL_VARIANT_DIRECTORY / f"{arguments.model_variant}.yaml")
    model = apply_stub_head_weights(build_glare_model(OmegaConf.to_container(variant_config.MODEL, resolve=True)))
    export_report = export_and_verify_glare_model(model, arguments.out, write_fp16_variant=False)
    print(json.dumps(export_report, indent=2))


if __name__ == "__main__":
    main()
