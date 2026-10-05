"""Entry point: `python -m glare_model.train --config configs/glare_model/base.yaml [KEY=VALUE ...]`.

Composes the config (see `training/config_loading.py`), trains (resuming automatically if the
hashed run directory already has `checkpoints/last.pt`), and on normal completion exports the
EMA weights to `<run>/exported/glare_removal.onnx` with the same checks as `export_onnx`.

Launch long runs politely on the shared machine:
    nice -n 10 python -m glare_model.train --config configs/glare_model/base.yaml CONFIG_GROUPS.DATA=real
"""

import argparse
import logging
from pathlib import Path

from glare_model.export.onnx_export import export_glare_model_to_onnx_model, save_onnx_model
from glare_model.training.config_loading import load_training_config
from glare_model.training.glare_model_trainer import GlareModelTrainer
from glare_model.training.optimization import apply_mps_memory_cap

logger = logging.getLogger(__name__)

EXPORTED_MODEL_FILENAME = "glare_removal.onnx"


def parse_command_line() -> argparse.Namespace:
    """Parse `--config` plus free-form dotlist overrides."""
    parser = argparse.ArgumentParser(description="Train the eyeglass glare removal model.")
    parser.add_argument("--config", type=Path, default=Path("configs/glare_model/base.yaml"))
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. SOLVER.MAX_ITER=1000")
    return parser.parse_args()


def main() -> None:
    """Train, then export the EMA model if training reached MAX_ITER."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line()
    logging.getLogger("onnx_ir").setLevel(logging.WARNING)
    training_config = load_training_config(arguments.config, arguments.overrides)
    apply_mps_memory_cap(training_config.TRAIN.MPS_HIGH_WATERMARK_RATIO, training_config.TRAIN.MPS_LOW_WATERMARK_RATIO)
    trainer = GlareModelTrainer(training_config)
    trainer.train()
    if trainer.step >= training_config.SOLVER.MAX_ITER and not trainer.stop_requested:
        exported_model = export_glare_model_to_onnx_model(trainer.ema.averaged_model)
        save_onnx_model(exported_model, trainer.run_directory.exported_directory / EXPORTED_MODEL_FILENAME)


if __name__ == "__main__":
    main()
