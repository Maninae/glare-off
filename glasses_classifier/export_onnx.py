"""Entry point: `python -m glasses_classifier.export_onnx --checkpoint <run>/checkpoints/best.pt [--out app/models/glasses_classifier.onnx]`.

Exports the checkpoint to the app contract (fp16 weights when parity holds), verifies it, and
prints the JSON report (params, size, op inventory, parity, single-thread CPU latency).
"""

import argparse
import json
import logging
from pathlib import Path

from glasses_classifier.checkpointing import load_glasses_classifier_checkpoint
from glasses_classifier.glasses_onnx_export import export_and_verify_glasses_classifier


def main() -> None:
    """Load, export, verify, print."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export a glasses-classifier checkpoint to the app's ONNX contract.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("app/models/glasses_classifier.onnx"))
    arguments = parser.parse_args()
    model, _ = load_glasses_classifier_checkpoint(arguments.checkpoint)
    print(json.dumps(export_and_verify_glasses_classifier(model, arguments.out), indent=2))


if __name__ == "__main__":
    main()
