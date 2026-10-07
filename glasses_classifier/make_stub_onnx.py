"""Entry point: `python -m glasses_classifier.make_stub_onnx [--out app/models/glasses_classifier.onnx]`.

Writes an UNTRAINED stub with the real architecture and the real I/O signature, so the app can be
wired before training finishes. The head bias is set large, so the stub says "glasses" (p ~ 0.99)
for every crop: the app then behaves as if the gate were absent. Replace it with
`python -m glasses_classifier.export_onnx` once a trained checkpoint exists.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from glasses_classifier.glasses_classifier_net import GlassesClassifierNet
from glasses_classifier.glasses_onnx_export import export_and_verify_glasses_classifier

STUB_SEED = 0
# sigmoid(5) = 0.993: every face passes the gate until the trained model lands.
STUB_HEAD_BIAS = 5.0


def parse_command_line() -> argparse.Namespace:
    """Parse the output path."""
    parser = argparse.ArgumentParser(description="Write an untrained always-glasses stub of the glasses classifier.")
    parser.add_argument("--out", type=Path, default=Path("app/models/glasses_classifier.onnx"))
    return parser.parse_args()


def main() -> None:
    """Build the default net, zero the head weights, set the bias, export, verify, print the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line()
    torch.manual_seed(STUB_SEED)
    model = GlassesClassifierNet().eval()
    with torch.no_grad():
        model.classifier_head.weight.zero_()
        model.classifier_head.bias.fill_(STUB_HEAD_BIAS)
    print(json.dumps(export_and_verify_glasses_classifier(model, arguments.out), indent=2))


if __name__ == "__main__":
    main()
