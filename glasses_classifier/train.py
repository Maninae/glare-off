"""Entry point: `python -m glasses_classifier.train [--config configs/glasses_classifier/base.yaml] [KEY=VALUE ...]`.

Trains GlassesClassifierNet on the cached eye crops (build them first with
`python -m glasses_classifier.build_eye_crop_cache`), then evaluates the best checkpoint and
exports it to `<run>/exported/glasses_classifier.onnx` (copy it into `app/models/` by hand, or run
`export_onnx` with `--out app/models/glasses_classifier.onnx`).

- Loss: BCE on logits. Class balance comes from a WeightedRandomSampler (both classes drawn equally),
  not a pos_weight, so the probabilities stay calibrated to a 50/50 prior.
- AdamW, per-step linear warmup then cosine (glare_model's scheduler), fp32.
- Best checkpoint = highest val ROC AUC, ties broken by lower val BCE (AUC is threshold-free but
  saturates at 1.0 on this small val set; ~57 bare faces make fixed-threshold accuracy coarse).
- Run directory: `OUTPUT.RUNS_ROOT/OUTPUT.RUN_NAME/{checkpoints,tb,evaluation,exported,config.yaml}`.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from glare_model.data.crop_augmentation import PhotometricJitterSettings
from glare_model.training.optimization import (
    apply_mps_memory_cap,
    build_warmup_cosine_scheduler,
    read_peak_accelerator_memory_gib,
    seed_everything,
    select_training_device,
)
from glasses_classifier.checkpointing import save_glasses_classifier_checkpoint
from glasses_classifier.classification_metrics import compute_binary_classification_report, compute_roc_auc
from glasses_classifier.config_loading import DEFAULT_CONFIG_PATH, load_glasses_classifier_config
from glasses_classifier.evaluate import evaluate_glasses_classifier, predict_glasses_probabilities
from glasses_classifier.glasses_classifier_net import GlassesClassifierNet, build_glasses_classifier_from_config, count_trainable_parameters
from glasses_classifier.glasses_crop_dataset import GlassesCropDataset
from glasses_classifier.glasses_onnx_export import export_and_verify_glasses_classifier

logger = logging.getLogger(__name__)


def build_training_loader(config: DictConfig) -> DataLoader:
    """Train DataLoader over the cached crops, class-balanced if configured."""
    photometric = config.AUGMENTATION.PHOTOMETRIC
    train_dataset = GlassesCropDataset(
        Path(config.CACHE.DIRECTORY),
        "train",
        is_training=True,
        photometric_settings=PhotometricJitterSettings(**{key.lower(): float(value) for key, value in photometric.items()}),
        grayscale_probability=float(config.AUGMENTATION.GRAYSCALE_PROBABILITY),
    )
    sampler = None
    if config.DATALOADER.CLASS_BALANCED_SAMPLING:
        sampler = WeightedRandomSampler(train_dataset.class_balanced_sample_weights(), num_samples=len(train_dataset), replacement=True)
    return DataLoader(
        train_dataset,
        batch_size=int(config.DATALOADER.BATCH_SIZE),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(config.DATALOADER.NUM_WORKERS),
        persistent_workers=int(config.DATALOADER.NUM_WORKERS) > 0,
        drop_last=True,
    )


def build_solver_node_for_scheduler(config: DictConfig, steps_per_epoch: int) -> DictConfig:
    """Translate the epoch-based SOLVER block into the iteration-based keys glare_model's scheduler reads."""
    return OmegaConf.create(
        {
            "BASE_LR": float(config.SOLVER.BASE_LR),
            "MIN_LR": float(config.SOLVER.MIN_LR),
            "WARMUP_ITERS": int(config.SOLVER.WARMUP_EPOCHS) * steps_per_epoch,
            "MAX_ITER": int(config.SOLVER.EPOCHS) * steps_per_epoch,
        }
    )


def validate_one_epoch(model: GlassesClassifierNet, val_dataset: GlassesCropDataset, device: torch.device, default_threshold: float) -> dict:
    """Return val BCE, ROC AUC, and accuracy / balanced accuracy / per-class recall at the default threshold."""
    probabilities = predict_glasses_probabilities(model, val_dataset, device)
    labels = val_dataset.has_glasses_labels
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    report = compute_binary_classification_report(probabilities, labels, default_threshold)
    return {
        "val_bce": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "val_roc_auc": compute_roc_auc(probabilities, labels),
        "val_accuracy": report.accuracy,
        "val_balanced_accuracy": report.balanced_accuracy,
        "val_glasses_recall": report.glasses_recall,
        "val_bare_recall": report.bare_recall,
    }


def train_glasses_classifier(config: DictConfig) -> Path:
    """Run the full training + evaluation + export; return the run directory."""
    run_directory = Path(config.OUTPUT.RUNS_ROOT) / config.OUTPUT.RUN_NAME
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_directory / "config.yaml")
    apply_mps_memory_cap(float(config.TRAIN.MPS_HIGH_WATERMARK_RATIO), float(config.TRAIN.MPS_LOW_WATERMARK_RATIO))
    seed_everything(int(config.GLOBAL.SEED))
    device = select_training_device(config.TRAIN.DEVICE)

    train_loader = build_training_loader(config)
    val_dataset = GlassesCropDataset(Path(config.CACHE.DIRECTORY), "val", is_training=False)
    model = build_glasses_classifier_from_config(OmegaConf.to_container(config.MODEL, resolve=True)).to(device)
    logger.info("device %s, %d parameters, %d train faces, %d steps/epoch", device, count_trainable_parameters(model), len(train_loader.dataset), len(train_loader))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.SOLVER.BASE_LR), weight_decay=float(config.SOLVER.WEIGHT_DECAY))
    scheduler = build_warmup_cosine_scheduler(optimizer, build_solver_node_for_scheduler(config, len(train_loader)))
    loss_function = nn.BCEWithLogitsLoss()
    tensorboard_writer = SummaryWriter(str(run_directory / "tb"))
    default_threshold = float(config.EVALUATION.DEFAULT_THRESHOLD)

    best_selection_key, global_step, training_start = (-1.0, 0.0), 0, time.time()
    for epoch in range(int(config.SOLVER.EPOCHS)):
        model.train()
        epoch_start = time.time()
        for batch in train_loader:
            logits = model.forward_logits(batch["eye_crop"].to(device))
            loss = loss_function(logits, batch["has_glasses_label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % int(config.TRAIN.LOG_PERIOD) == 0:
                tensorboard_writer.add_scalar("train/bce", loss.item(), global_step)
                tensorboard_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                logger.info("epoch %d step %d bce %.4f lr %.2e peak %.2f GiB", epoch, global_step, loss.item(), scheduler.get_last_lr()[0], read_peak_accelerator_memory_gib(device))

        val_metrics = validate_one_epoch(model, val_dataset, device, default_threshold)
        val_metrics["epoch_seconds"] = time.time() - epoch_start
        for metric_name, metric_value in val_metrics.items():
            tensorboard_writer.add_scalar(f"val/{metric_name}", metric_value, epoch)
        logger.info("epoch %d val %s", epoch, json.dumps({key: round(value, 4) for key, value in val_metrics.items()}))
        save_glasses_classifier_checkpoint(model, config, epoch, val_metrics, checkpoint_directory / "last.pt")
        selection_key = (val_metrics["val_roc_auc"], -val_metrics["val_bce"])
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            save_glasses_classifier_checkpoint(model, config, epoch, val_metrics, checkpoint_directory / "best.pt")
            logger.info("new best (val AUC %.4f, val BCE %.4f) at epoch %d", selection_key[0], -selection_key[1], epoch)
    logger.info("training took %.1f min", (time.time() - training_start) / 60.0)
    tensorboard_writer.close()
    # Shut the persistent workers down now: left to interpreter teardown they abort the process (libc++ mutex error).
    del train_loader

    best_model = build_glasses_classifier_from_config(OmegaConf.to_container(config.MODEL, resolve=True))
    best_model.load_state_dict(torch.load(checkpoint_directory / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    best_model.eval()
    evaluation_report = evaluate_glasses_classifier(
        best_model, Path(config.CACHE.DIRECTORY), float(config.EVALUATION.MIN_GLASSES_RECALL), default_threshold, run_directory / "evaluation", torch.device("cpu")
    )
    logger.info("evaluation: %s", json.dumps({split: values["at_chosen_threshold"] for split, values in evaluation_report["splits"].items()}))
    export_report = export_and_verify_glasses_classifier(best_model, run_directory / "exported" / "glasses_classifier.onnx")
    with open(run_directory / "exported" / "export_report.json", "w") as report_file:
        json.dump(export_report, report_file, indent=2)
    return run_directory


def main() -> None:
    """Parse the config path and dotlist overrides, then train."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the glasses classifier on cached eye crops.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE dotlist overrides")
    arguments = parser.parse_args()
    train_glasses_classifier(load_glasses_classifier_config(arguments.config, arguments.overrides))


if __name__ == "__main__":
    main()
