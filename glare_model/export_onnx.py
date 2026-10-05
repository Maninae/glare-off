"""Entry point: `python -m glare_model.export_onnx --checkpoint <run>/checkpoints/best_ema.pt --out app/models/glare_removal.onnx`.

Rebuilds the model from the config stored in the checkpoint, loads the EMA weights (pass
`--live-weights` for the raw optimizer weights), exports per the Model I/O contract, and prints a
JSON report: parameter count, file sizes, op inventory, parity at 256x512 and 512x1024, fp16
variant, and single-thread CPU latency.
"""

import argparse
import json
import logging
from pathlib import Path

from glare_model.export.export_and_verify import export_and_verify_glare_model
from glare_model.training.checkpointing import load_model_weights_for_inference
from glare_model.training.component_builders import build_glare_model

logger = logging.getLogger(__name__)


def parse_command_line() -> argparse.Namespace:
    """Parse checkpoint, output path, and variant flags."""
    parser = argparse.ArgumentParser(description="Export a trained glare model checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("app/models/glare_removal.onnx"))
    parser.add_argument("--live-weights", action="store_true", help="export live weights instead of EMA")
    parser.add_argument("--no-fp16", action="store_true", help="skip the fp16-weights variant")
    return parser.parse_args()


def main() -> None:
    """Export, verify, and print the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line()
    model_weights, model_config = load_model_weights_for_inference(arguments.checkpoint, prefer_ema=not arguments.live_weights)
    model = build_glare_model(model_config)
    model.load_state_dict(model_weights)
    export_report = export_and_verify_glare_model(model, arguments.out, write_fp16_variant=not arguments.no_fp16)
    print(json.dumps(export_report, indent=2))


if __name__ == "__main__":
    main()
