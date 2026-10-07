"""Entry point: `python -m glasses_classifier.evaluate --checkpoint <run>/checkpoints/best.pt`.

Scores a checkpoint on val, test, and real_glare_eval, chooses the threshold on val, and writes:
- `<run>/evaluation/evaluation_report.json`: AUC + reports at the chosen threshold and at 0.5, per split.
- `<run>/evaluation/misclassified__test_and_real_glare_eval.png`: contact sheet of every wrong face.

- real_glare_eval faces all wear glasses, so they only measure glasses recall (extra positives, never trained on).
- Threshold rule (`classification_metrics.choose_threshold_for_glasses_recall`): EVALUATION.DEFAULT_THRESHOLD
  unless val glasses recall there is below EVALUATION.MIN_GLASSES_RECALL; then the highest lower grid value meeting it.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from glasses_classifier.checkpointing import load_glasses_classifier_checkpoint
from glasses_classifier.classification_metrics import (
    choose_threshold_for_glasses_recall,
    compute_binary_classification_report,
    compute_roc_auc,
)
from glasses_classifier.contact_sheet import write_eye_crop_contact_sheet
from glasses_classifier.glasses_classifier_net import GlassesClassifierNet
from glasses_classifier.glasses_crop_dataset import GlassesCropDataset

logger = logging.getLogger(__name__)

EVALUATION_SPLITS = ("val", "test", "real_glare_eval")
THRESHOLD_SELECTION_SPLIT = "val"
MISCLASSIFIED_SHEET_SPLITS = ("test", "real_glare_eval")
EVALUATION_BATCH_SIZE = 32


@torch.no_grad()
def predict_glasses_probabilities(model: GlassesClassifierNet, dataset: GlassesCropDataset, device: torch.device) -> np.ndarray:
    """Return the [N] glasses probabilities of every face in `dataset`, in index order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False, num_workers=0)
    probabilities = [model(batch["eye_crop"].to(device)).squeeze(1).cpu().numpy() for batch in loader]
    return np.concatenate(probabilities)


def evaluate_glasses_classifier(
    model: GlassesClassifierNet, cache_directory: Path, min_glasses_recall: float, default_threshold: float, output_directory: Path, device: torch.device
) -> dict:
    """Score every evaluation split, choose the threshold on val, write the report JSON and the misclassified sheet; return the report."""
    model = model.to(device)
    datasets = {split: GlassesCropDataset(cache_directory, split, is_training=False) for split in EVALUATION_SPLITS}
    probabilities = {split: predict_glasses_probabilities(model, dataset, device) for split, dataset in datasets.items()}
    chosen_threshold = choose_threshold_for_glasses_recall(
        probabilities[THRESHOLD_SELECTION_SPLIT], datasets[THRESHOLD_SELECTION_SPLIT].has_glasses_labels, min_glasses_recall, default_threshold
    )
    evaluation_report: dict = {"chosen_threshold": chosen_threshold, "min_glasses_recall_on_val": min_glasses_recall, "splits": {}}
    for split, dataset in datasets.items():
        labels = dataset.has_glasses_labels
        evaluation_report["splits"][split] = {
            "roc_auc": compute_roc_auc(probabilities[split], labels),
            "at_chosen_threshold": compute_binary_classification_report(probabilities[split], labels, chosen_threshold).to_dict(),
            "at_default_threshold": compute_binary_classification_report(probabilities[split], labels, default_threshold).to_dict(),
        }

    misclassified_crops, misclassified_captions, misclassified_rows = [], [], []
    for split in MISCLASSIFIED_SHEET_SPLITS:
        dataset = datasets[split]
        wrong_indices = np.flatnonzero((probabilities[split] >= chosen_threshold) != (dataset.has_glasses_labels >= 0.5))
        for face_index in wrong_indices:
            source_id = dataset.face_rows[face_index]["source_id"]
            truth = "glasses" if dataset.has_glasses_labels[face_index] >= 0.5 else "bare"
            misclassified_crops.append(dataset.read_crop_uint8(int(face_index)))
            misclassified_captions.append(f"{split} {source_id} true={truth} p={probabilities[split][face_index]:.2f}")
            misclassified_rows.append({"split": split, "source_id": source_id, "true_class": truth, "glasses_probability": float(probabilities[split][face_index])})
    evaluation_report["misclassified"] = misclassified_rows

    output_directory.mkdir(parents=True, exist_ok=True)
    sheet_path = write_eye_crop_contact_sheet(misclassified_crops, misclassified_captions, output_directory / "misclassified__test_and_real_glare_eval.png")
    evaluation_report["misclassified_sheet"] = str(sheet_path)
    with open(output_directory / "evaluation_report.json", "w") as report_file:
        json.dump(evaluation_report, report_file, indent=2)
    return evaluation_report


def main() -> None:
    """Evaluate a checkpoint on CPU and print the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate a glasses-classifier checkpoint (val threshold, test report, misclassified sheet).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    arguments = parser.parse_args()
    model, config = load_glasses_classifier_checkpoint(arguments.checkpoint)
    evaluation_report = evaluate_glasses_classifier(
        model,
        Path(config["CACHE"]["DIRECTORY"]),
        float(config["EVALUATION"]["MIN_GLASSES_RECALL"]),
        float(config["EVALUATION"]["DEFAULT_THRESHOLD"]),
        arguments.checkpoint.parent.parent / "evaluation",
        torch.device("cpu"),
    )
    print(json.dumps(evaluation_report, indent=2))


if __name__ == "__main__":
    main()
