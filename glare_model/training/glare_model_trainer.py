"""GlareModelTrainer: the coordinator that owns all training state and runs the step loop.

State lives here (model, EMA, optimizer, scheduler, step); the helpers it calls are stateless.
The loop is iteration-based (`SOLVER.MAX_ITER`), not epoch-based, because the data is synthesized
on the fly and an "epoch" has no meaning beyond one pass over the source faces.

One optimizer step = `SOLVER.GRADIENT_ACCUMULATION_STEPS` micro-batches of `DATALOADER.BATCH_SIZE`,
so the effective batch fits the shared machine's memory cap (no BatchNorm, so accumulation is
exact). Plain fp32: autocast on MPS measured only ~7% faster, so it is not offered.

Interruption: SIGINT / SIGTERM set a flag; the loop finishes the current step, writes
`checkpoints/last.pt`, and returns. Rerunning the same command resumes from it (`TRAIN.RESUME`).
"""

import logging
import math
import signal
import time
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from glare_model.training.checkpointing import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint_atomically,
)
from glare_model.training.component_builders import (
    TRAINING_SPLIT,
    VALIDATION_SPLIT,
    build_data_loader,
    build_glare_model,
    build_glare_pair_dataset,
    build_glare_removal_loss,
    config_section_as_dict,
)
from glare_model.training.optimization import (
    ExponentialMovingAverageModel,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    read_peak_accelerator_memory_gib,
    seed_everything,
    select_training_device,
)
from glare_model.training.run_directory import prepare_run_directory
from glare_model.training.validation_pass import move_batch_to_device, run_validation_pass

logger = logging.getLogger(__name__)

BEST_METRIC_NAME = "psnr_lens"


class GlareModelTrainer:
    """Builds every training component from the config and runs train / validate / checkpoint."""

    def __init__(self, training_config: DictConfig):
        self.training_config = training_config
        seed_everything(training_config.GLOBAL.SEED)
        self.device = select_training_device(training_config.TRAIN.DEVICE)
        self.run_directory = prepare_run_directory(training_config)

        self.model = build_glare_model(training_config.MODEL).to(self.device)
        self.ema = ExponentialMovingAverageModel(self.model, training_config.SOLVER.EMA_DECAY)
        self.optimizer = build_adamw_optimizer(self.model, training_config.SOLVER)
        self.scheduler = build_warmup_cosine_scheduler(self.optimizer, training_config.SOLVER)
        self.loss_function = build_glare_removal_loss(training_config.LOSS)

        self.gradient_accumulation_steps = training_config.SOLVER.GRADIENT_ACCUMULATION_STEPS
        self.peak_accelerator_memory_gib = 0.0

        self.training_loader = build_data_loader(build_glare_pair_dataset(training_config, TRAINING_SPLIT), training_config, shuffle=True)
        self.validation_loader = build_data_loader(build_glare_pair_dataset(training_config, VALIDATION_SPLIT), training_config, shuffle=False)
        self.summary_writer = SummaryWriter(str(self.run_directory.tensorboard_directory))

        self.step = 0
        self.best_metric_value = -math.inf
        self.stop_requested = False
        if training_config.TRAIN.RESUME:
            self.resume_from_last_checkpoint()
        logger.info("device %s, starting at step %d of %d", self.device, self.step, training_config.SOLVER.MAX_ITER)

    def resume_from_last_checkpoint(self) -> None:
        """Load `checkpoints/last.pt` if it exists (no-op on a fresh run)."""
        last_checkpoint_path = self.run_directory.checkpoint_directory / LAST_CHECKPOINT_NAME
        if not last_checkpoint_path.exists():
            return
        checkpoint_state = load_checkpoint(last_checkpoint_path)
        self.model.load_state_dict(checkpoint_state["model"])
        self.ema.averaged_model.load_state_dict(checkpoint_state["ema_model"])
        self.optimizer.load_state_dict(checkpoint_state["optimizer"])
        self.scheduler.load_state_dict(checkpoint_state["scheduler"])
        restore_rng_states(checkpoint_state["rng_states"])
        self.step = checkpoint_state["step"]
        self.best_metric_value = checkpoint_state["best_metric_value"]
        logger.info("resumed from %s at step %d", last_checkpoint_path, self.step)

    def build_checkpoint_state(self) -> dict[str, Any]:
        """Return everything needed to resume, plus the model config needed to rebuild for export."""
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema_model": self.ema.averaged_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_states": capture_rng_states(),
            "best_metric_value": self.best_metric_value,
            "model_config": config_section_as_dict(self.training_config.MODEL),
            "training_config": OmegaConf.to_container(self.training_config, resolve=True),
        }

    def save_last_checkpoint(self) -> None:
        """Atomically write `checkpoints/last.pt`."""
        save_checkpoint_atomically(self.build_checkpoint_state(), self.run_directory.checkpoint_directory / LAST_CHECKPOINT_NAME)

    def request_stop(self, signal_number: int, frame: Any) -> None:
        """Signal handler: finish the current step, checkpoint, and stop."""
        logger.warning("signal %d received: checkpointing after this step", signal_number)
        self.stop_requested = True

    def iterate_training_batches(self):
        """Yield training batches forever, re-iterating the loader at each pass end."""
        while True:
            yield from self.training_loader

    def run_training_step(self, micro_batches: list[dict]) -> dict[str, torch.Tensor]:
        """One optimizer step over `micro_batches`; returns mean loss terms and the pre-clip grad norm."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        summed_loss_outputs: dict[str, torch.Tensor] = {}
        for micro_batch in micro_batches:
            micro_batch = move_batch_to_device(micro_batch, self.device)
            restoration_delta, mask_logits = self.model.predict_delta_and_mask_logits(micro_batch["glare_eye_crop"])
            loss_outputs = self.loss_function(restoration_delta, mask_logits, micro_batch)
            total_loss = loss_outputs["loss_total"]
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss {total_loss.item()} at step {self.step}")
            (total_loss / len(micro_batches)).backward()
            for name, value in loss_outputs.items():
                summed_loss_outputs[name] = summed_loss_outputs.get(name, 0.0) + value.detach() / len(micro_batches)
            self.peak_accelerator_memory_gib = max(self.peak_accelerator_memory_gib, read_peak_accelerator_memory_gib(self.device))

        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.SOLVER.GRAD_CLIP_NORM)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.model)
        return summed_loss_outputs | {"gradient_norm": gradient_norm.detach()}

    def iterate_optimizer_step_micro_batches(self):
        """Yield lists of `gradient_accumulation_steps` training micro-batches, forever."""
        micro_batches = []
        for micro_batch in self.iterate_training_batches():
            micro_batches.append(micro_batch)
            if len(micro_batches) == self.gradient_accumulation_steps:
                yield micro_batches
                micro_batches = []

    def validate_and_log(self) -> dict[str, float]:
        """Validate the EMA model, log scalars and the image grid, keep the best EMA checkpoint."""
        train_config = self.training_config.TRAIN
        metrics, image_grid = run_validation_pass(
            self.ema.averaged_model, self.validation_loader, self.device, train_config.IMAGE_GRID_COUNT, train_config.MAX_VAL_BATCHES
        )
        for metric_name, metric_value in metrics.items():
            self.summary_writer.add_scalar(f"val/{metric_name}", metric_value, self.step)
        if image_grid is not None:
            self.summary_writer.add_image("val/input_pred_target_maskpred_masktarget_lostpred", image_grid, self.step)
        logger.info("step %d val %s", self.step, " ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
        if metrics[BEST_METRIC_NAME] > self.best_metric_value:
            self.best_metric_value = metrics[BEST_METRIC_NAME]
            save_checkpoint_atomically(self.build_checkpoint_state(), self.run_directory.checkpoint_directory / BEST_CHECKPOINT_NAME)
        return metrics

    def train(self) -> dict[str, float]:
        """Run steps until MAX_ITER or a stop signal; return the last validation metrics."""
        train_config = self.training_config.TRAIN
        max_iter = self.training_config.SOLVER.MAX_ITER
        previous_handlers = {sig: signal.signal(sig, self.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        last_metrics: dict[str, float] = {}
        window_start_time = time.perf_counter()
        window_start_step = self.step
        try:
            for micro_batches in self.iterate_optimizer_step_micro_batches():
                if self.step >= max_iter or self.stop_requested:
                    break
                step_outputs = self.run_training_step(micro_batches)
                self.step += 1

                if self.step % train_config.LOG_PERIOD == 0 or self.step == max_iter:
                    seconds_per_step = (time.perf_counter() - window_start_time) / max(self.step - window_start_step, 1)
                    for output_name, output_value in step_outputs.items():
                        self.summary_writer.add_scalar(f"train/{output_name}", output_value.item(), self.step)
                    self.summary_writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.step)
                    self.summary_writer.add_scalar("train/seconds_per_step", seconds_per_step, self.step)
                    self.summary_writer.add_scalar("train/peak_accelerator_memory_gib", self.peak_accelerator_memory_gib, self.step)
                    logger.info(
                        "step %d/%d loss %.4f lens_l1 %.4f focal %.4f lr %.2e %.3f s/step peak %.2f GiB",
                        self.step, max_iter, step_outputs["loss_total"].item(), step_outputs["loss_lens_l1"].item(),
                        step_outputs["loss_mask_focal"].item(), self.scheduler.get_last_lr()[0], seconds_per_step,
                        self.peak_accelerator_memory_gib,
                    )
                    window_start_time, window_start_step = time.perf_counter(), self.step

                is_validation_step = self.step % train_config.VAL_PERIOD == 0 or self.step == max_iter
                is_checkpoint_step = self.step % train_config.CHECKPOINT_PERIOD == 0 or self.step == max_iter
                if is_validation_step:
                    last_metrics = self.validate_and_log()
                if is_checkpoint_step:
                    self.save_last_checkpoint()
                if is_validation_step or is_checkpoint_step:
                    # Keep s/step a pure training-step measurement.
                    window_start_time, window_start_step = time.perf_counter(), self.step
        finally:
            if self.stop_requested:
                self.save_last_checkpoint()
                logger.info("stopped at step %d; rerun the same command to resume", self.step)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            self.summary_writer.flush()
        return last_metrics
