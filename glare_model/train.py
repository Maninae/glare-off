"""Entry point: `python -m glare_model.train --config configs/glare_model/base.yaml [KEY=VALUE ...]`.

Composes the config (see `training/config_loading.py`), trains (resuming automatically if the
hashed run directory already has `checkpoints/last.pt`), and on normal completion exports
`checkpoints/best_ema.pt` (the best EMA weights under the selection rule in
`training/validation_metrics.py`) to `<run>/exported/glare_removal.onnx`. If no checkpoint ever
qualified as best, it exports the final EMA weights and says so.

Launch long runs politely on the shared machine:
    nice -n 10 python -m glare_model.train --config configs/glare_model/base.yaml CONFIG_GROUPS.DATA=real
"""

import argparse
import logging
from pathlib import Path

from torch import nn

from glare_model.export.onnx_export import export_glare_model_to_onnx_model, save_onnx_model
from glare_model.training.checkpointing import BEST_CHECKPOINT_NAME, load_model_weights_for_inference
from glare_model.training.component_builders import build_glare_model
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


def select_model_to_export(trainer: GlareModelTrainer) -> nn.Module:
    """Return the best EMA model from `checkpoints/best_ema.pt`, or the final EMA model if none exists."""
    best_checkpoint_path = trainer.run_directory.checkpoint_directory / BEST_CHECKPOINT_NAME
    if not best_checkpoint_path.exists():
        logger.warning("no %s (no checkpoint passed selection); exporting the final EMA weights", BEST_CHECKPOINT_NAME)
        return trainer.ema.averaged_model
    model_weights, model_config = load_model_weights_for_inference(best_checkpoint_path, prefer_ema=True)
    best_model = build_glare_model(model_config)
    best_model.load_state_dict(model_weights)
    return best_model.eval()


def main() -> None:
    """Train, then export the best EMA model if training reached MAX_ITER."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arguments = parse_command_line()
    logging.getLogger("onnx_ir").setLevel(logging.WARNING)
    training_config = load_training_config(arguments.config, arguments.overrides)
    apply_mps_memory_cap(training_config.TRAIN.MPS_HIGH_WATERMARK_RATIO, training_config.TRAIN.MPS_LOW_WATERMARK_RATIO)
    trainer = GlareModelTrainer(training_config)
    trainer.train()
    if trainer.step >= training_config.SOLVER.MAX_ITER and not trainer.stop_requested:
        exported_model = export_glare_model_to_onnx_model(select_model_to_export(trainer))
        save_onnx_model(exported_model, trainer.run_directory.exported_directory / EXPORTED_MODEL_FILENAME)


if __name__ == "__main__":
    main()
