"""Train a Contour Proposal Network on the HAICU bubble COCO dataset.

The structure mirrors ``demos/code/mydata_uncertainty.py``. Dataset-specific
code is delegated to ``bubble_coco_dataset.py``.
"""

from __future__ import annotations

import atexit
import heapq
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import celldetection as cd
from bubble_coco_dataset import (
    DEFAULT_DATA_ROOT,
    MIN_VISIBLE_AREA_FRACTION,
    CaseBatchSampler,
    HAICUBubbleCPNDataset,
)


DEFAULT_TRAIN_CASES = [case_id for case_id in range(1, 26) if case_id not in {5, 10, 11}]
DEFAULT_VALIDATION_CASES = [5, 10, 11]
IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
VISUALIZATION_COUNT = 10
VISUALIZATION_SEED = 42


def environment_case_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    return default if value is None else [int(case_id) for case_id in value.split(",") if case_id]


TRAIN_CASES = environment_case_list("HAICU_TRAIN_CASES", DEFAULT_TRAIN_CASES)
VALIDATION_CASES = environment_case_list("HAICU_VALIDATION_CASES", DEFAULT_VALIDATION_CASES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


conf = cd.Config(
    uncertainty_head=True,
    uncertainty_nms=True,
    uncertainty_factor=7.0,
    directory=str(DEFAULT_DATA_ROOT),
    download_data=False,
    in_channels=1,
    classes=2,
    shuffle=True,
    bg_fg_dists=(0.8, 0.85),
    crop_size=None,
    # cpn="CpnResNet50UNet",
    cpn="CpnU22",
    score_thresh=0.9,
    # val_score_threshs=(0.6, 0.8, 0.9),
    val_score_threshs=(0.9,),
    nms_thresh=0.5,
    # val_nms_threshs=(0.3, 0.5, 0.8),
    val_nms_threshs=(0.5,),
    contour_head_stride=2,
    order=6,
    samples=64,
    refinement_iterations=3,
    refinement_buckets=6,
    inputs_mean=0.5,
    inputs_std=0.5,
    tweaks={"BatchNorm2d": {"momentum": 0.1}},
    optimizer={"Adam": {"lr": 0.0001, "betas": (0.9, 0.999), "weight_decay": 1e-3}},
    scheduler={"StepLR": {"step_size": 5, "gamma": 0.5}},
    epochs=int(os.environ.get("HAICU_EPOCHS", "20")),
    batch_size=1,
    amp=torch.cuda.is_available(),
    test_batch_size=1,
    num_workers=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    min_visible_area_fraction=MIN_VISIBLE_AREA_FRACTION,
)
print(conf)


output_root = Path(__file__).resolve().parents[1] / "outputs"
tracking_root = output_root / "mlruns"
tracking_root.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{tracking_root / 'mlflow.db'}"))
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "haicu-bubble-cpn")
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name, artifact_location=(tracking_root / "artifacts").resolve().as_uri())
mlflow.set_experiment(experiment_name)
mlflow_run = mlflow.start_run(run_name=f"haicu-coco-{datetime.now():%Y%m%d-%H%M%S}")
run_finished = False


def finish_mlflow_run(status: str = "FINISHED") -> None:
    global run_finished
    if not run_finished and mlflow.active_run() is not None:
        mlflow.end_run(status=status)
        run_finished = True


def mlflow_exception_hook(exc_type, exc_value, exc_traceback) -> None:
    finish_mlflow_run("FAILED")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = mlflow_exception_hook
atexit.register(finish_mlflow_run)

output_dir = output_root / f"run_{datetime.now():%Y%m%d_%H%M%S}_{mlflow_run.info.run_id[:8]}"
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir = output_dir / "figures"
figure_dir.mkdir()
prediction_dir = output_dir / "validation_predictions"
uncertainty_dir = output_dir / "uncertainty"
top_uncertainty_dir = uncertainty_dir / "top_instances"
config_path = output_dir / "config.json"
conf.to_json(str(config_path))
(output_dir / "split.json").write_text(json.dumps({"training_cases": TRAIN_CASES, "validation_cases": VALIDATION_CASES}, indent=2))
mlflow.log_params(
    {
        "data_root": str(DEFAULT_DATA_ROOT),
        "train_cases": ",".join(map(str, TRAIN_CASES)),
        "validation_cases": ",".join(map(str, VALIDATION_CASES)),
        "min_visible_area_fraction": conf.min_visible_area_fraction,
        "batch_size": conf.batch_size,
        "epochs": conf.epochs,
        "cpn": conf.cpn,
        "order": conf.order,
        "samples": conf.samples,
    }
)


def make_dataset(case_ids: list[int]) -> HAICUBubbleCPNDataset:
    return HAICUBubbleCPNDataset(
        DEFAULT_DATA_ROOT,
        case_ids,
        samples=conf.samples,
        order=conf.order,
        max_bg_dist=conf.bg_fg_dists[0],
        min_fg_dist=conf.bg_fg_dists[1],
        min_visible_area_fraction=conf.min_visible_area_fraction,
    )


train_data = make_dataset(TRAIN_CASES)
val_data = make_dataset(VALIDATION_CASES)
train_sampler = CaseBatchSampler(train_data, conf.batch_size, seed=42)
train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=conf.num_workers, collate_fn=cd.universal_dict_collate_fn)
val_loader = DataLoader(val_data, batch_size=conf.test_batch_size, num_workers=conf.num_workers, collate_fn=cd.universal_dict_collate_fn)
print(f"Training set: {len(train_data)} images")
print(f"Validation set: {len(val_data)} images")


def output_stem_for_record(record: dict) -> str:
    """Preserve the original COCO path in output filenames for easy lookup."""
    return "_".join(Path(record["file_name"]).with_suffix("").parts)


def labels_to_contours(labels: np.ndarray) -> list[np.ndarray]:
    contours = cd.data.labels2contours(labels)
    if isinstance(contours, dict):
        contours = contours.values()
    return [np.squeeze(contour, axis=1) if contour.ndim == 3 else contour for contour in contours]


def show_target_example(dataset: HAICUBubbleCPNDataset, filename: str, title: str) -> None:
    record, image, _ = dataset.raw_sample(0)
    sample = dataset[0]
    figure, axis = plt.subplots(figsize=(5, 12))
    axis.imshow(image, cmap="gray")
    for contour in labels_to_contours(sample["targets"]):
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], linewidth=0.7)
    axis.set(title=f"{title}: {record['file_name']}")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(figure_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def order_plot(dataset: HAICUBubbleCPNDataset) -> None:
    original_order = dataset.target_generator.order
    figure, axes = plt.subplots(2, 3, figsize=(15, 16))
    for order, axis in enumerate(axes.flat, start=1):
        dataset.target_generator.order = order
        record, image, _ = dataset.raw_sample(0)
        sample = dataset[0]
        axis.imshow(image, cmap="gray")
        for contour in sample["sampled_contours"][0]:
            closed = np.vstack((contour, contour[:1]))
            axis.plot(closed[:, 0], closed[:, 1], linewidth=0.5)
        axis.set(title=f"Order {order}: {record['file_name']}")
        axis.set_axis_off()
    dataset.target_generator.order = original_order
    figure.tight_layout()
    figure.savefig(figure_dir / "validation_order_plot.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


show_target_example(train_data, "train_target_example.png", "Training CPN target")
show_target_example(val_data, "validation_target_example.png", "Validation CPN target")
order_plot(val_data)


model = getattr(cd.models, conf.cpn)(
    in_channels=conf.in_channels,
    order=conf.order,
    samples=conf.samples,
    refinement_iterations=conf.refinement_iterations,
    nms_thresh=conf.nms_thresh,
    score_thresh=conf.score_thresh,
    contour_head_stride=conf.contour_head_stride,
    classes=conf.classes,
    refinement_buckets=conf.refinement_buckets,
    uncertainty_head=conf.uncertainty_head,
    uncertainty_nms=conf.uncertainty_nms,
    uncertainty_factor=conf.uncertainty_factor,
    backbone_kwargs={"inputs_mean": conf.inputs_mean, "inputs_std": conf.inputs_std},
).to(conf.device)
cd.conf2tweaks_(conf.tweaks, model)
optimizer = cd.conf2optimizer(conf.optimizer, model.parameters())
scheduler = cd.conf2scheduler(conf.scheduler, optimizer)
scaler = GradScaler() if conf.amp else None


def train_epoch(epoch: int) -> float:
    model.train()
    train_sampler.set_epoch(epoch)
    losses = []
    for batch in tqdm(train_loader, desc=f"Train {epoch}/{conf.epochs}"):
        batch = cd.to_device(batch, conf.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            loss = model(batch["inputs"], targets=batch)["loss"]
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(cd.asnumpy(loss)))
    return float(np.mean(losses))


def validate_epoch() -> float:
    was_training = model.training
    model.eval()
    model.training = True
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation loss", leave=False):
            batch = cd.to_device(batch, conf.device)
            with autocast(enabled=scaler is not None):
                losses.append(float(cd.asnumpy(model(batch["inputs"], targets=batch)["loss"])))
    model.train(was_training)
    return float(np.mean(losses))


def plot_loss_curves(train_losses: list[float], validation_losses: list[float]) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    for values, label, filename in ((train_losses, "Training loss", "train_loss_curve.png"), (validation_losses, "Validation loss", "validation_loss_curve.png")):
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(epochs, values, "-o", label=label)
        axis.set(xlabel="Epoch", ylabel="Loss", title=label)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=180)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(epochs, train_losses, "-o", label="Training loss")
    axis.plot(epochs, validation_losses, "-o", label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "loss_curves.png", dpi=180)
    plt.close(figure)


def save_prediction(image, target, contours, scores, uncertainties, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 12))
    axis.imshow(image, cmap="gray")
    for contour in labels_to_contours(target):
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color="white", linewidth=0.5, alpha=0.75)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(contours))))
    mean_uncertainties = None if uncertainties is None else np.asarray(uncertainties).mean(axis=1)
    for index, contour in enumerate(contours):
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=colors[index], linewidth=0.8)
        label = f"S={scores[index]:.2f}"
        if mean_uncertainties is not None:
            label += f"\nU={mean_uncertainties[index]:.2f}"
        center = contour.mean(axis=0)
        axis.text(center[0], center[1], label, color="white", fontsize=3.2, ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.2})
    axis.set(title=title)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def show_results(epoch: int) -> None:
    model.eval()
    batch = cd.to_device(next(iter(val_loader)), conf.device)
    with torch.no_grad():
        outputs = cd.asnumpy(model(batch["inputs"]))
    numpy_batch = cd.asnumpy(batch)
    image = cd.data.channels_first2channels_last(numpy_batch["inputs"][0])[..., 0]
    target = cd.data.channels_first2channels_last(numpy_batch["targets"][0])
    save_prediction(image, target, outputs["contours"][0], outputs["scores"][0], outputs.get("box_uncertainties", [None])[0], figure_dir / f"validation_prediction_epoch_{epoch:03d}.png", f"Validation prediction at epoch {epoch} | white: ground truth")


def save_uncertainty_overlay(
    image: np.ndarray,
    contours: np.ndarray,
    boxes: np.ndarray,
    uncertainties: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save one complete image with uncertainty boxes for every prediction."""
    figure, axis = plt.subplots(figsize=(5, 12))
    axis.imshow(image, cmap="gray")
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(contours))))
    text_style = {
        "color": "white",
        "fontsize": 3.2,
        "ha": "center",
        "va": "center",
        "bbox": {"facecolor": "black", "alpha": 0.55, "pad": 0.2},
    }
    for index, (contour, box, uncertainty, score) in enumerate(zip(contours, boxes, uncertainties, scores)):
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=colors[index], linewidth=0.7)
        x0, y0, x1, y1 = np.asarray(box).astype(int)
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=0.7))
        left, top, right, bottom = np.asarray(uncertainty)
        axis.text((x0 + x1) / 2, y0, f"{top * 100:.0f}%", **text_style)
        axis.text((x0 + x1) / 2, y1, f"{bottom * 100:.0f}%", **text_style)
        axis.text(x0, (y0 + y1) / 2, f"{left * 100:.0f}%", rotation=90, **text_style)
        axis.text(x1, (y0 + y1) / 2, f"{right * 100:.0f}%", rotation=90, **text_style)
        axis.text((x0 + x1) / 2, (y0 + y1) / 2, f"S={score:.2f}", **text_style)
    axis.set(title=title)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(save_predictions: bool = False):
    model.eval()
    results = cd.data.LabelMatcherList()
    candidate_heap = []
    candidate_counter = 0
    if save_predictions:
        visualization_rng = np.random.default_rng(VISUALIZATION_SEED)
        visualization_count = min(VISUALIZATION_COUNT, len(val_data))
        visualization_indices = set(
            visualization_rng.choice(len(val_data), size=visualization_count, replace=False).tolist()
        )
        print(f"Saving visualizations for {visualization_count} randomly selected validation images.")
    else:
        visualization_indices = set()

    def keep_uncertainty_candidate(value, image, contour, box, uncertainty, score):
        """Keep only the ten highest-uncertainty instances to bound RAM use."""
        nonlocal candidate_counter
        candidate_counter += 1
        if len(candidate_heap) >= 10 and float(value) <= candidate_heap[0][0]:
            return
        candidate = (
            float(value),
            image.copy(),
            np.asarray(contour).copy(),
            np.asarray(box).copy(),
            np.asarray(uncertainty).copy(),
            float(score),
        )
        entry = (candidate[0], candidate_counter, candidate)
        if len(candidate_heap) < 10:
            heapq.heappush(candidate_heap, entry)
        elif entry[0] > candidate_heap[0][0]:
            heapq.heapreplace(candidate_heap, entry)

    if save_predictions:
        prediction_dir.mkdir(exist_ok=True)
        uncertainty_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for sample_index, batch in enumerate(tqdm(val_loader, desc="Validation metrics")):
            record = val_data.records[sample_index]
            output_stem = output_stem_for_record(record)
            source_name = record["file_name"]
            device_batch = cd.to_device(batch, conf.device)
            outputs = cd.asnumpy(model(device_batch["inputs"]))
            numpy_batch = cd.asnumpy(batch)
            image = cd.data.channels_first2channels_last(numpy_batch["inputs"][0])[..., 0]
            target = cd.data.channels_first2channels_last(numpy_batch["targets"][0])
            contours = outputs["contours"][0]
            scores = outputs["scores"][0]
            uncertainties = outputs.get("box_uncertainties", [None])[0]
            boxes = outputs.get("boxes", [None])[0]
            prediction = cd.data.contours2labels(contours, target.shape[:2])
            results.append(cd.data.LabelMatcher(prediction, target))
            save_visualization = save_predictions and sample_index in visualization_indices
            if save_visualization:
                save_prediction(
                    image,
                    target,
                    contours,
                    scores,
                    uncertainties,
                    prediction_dir / f"{output_stem}.png",
                    f"Validation prediction | {source_name} | white: ground truth",
                )
                if uncertainties is not None and boxes is not None:
                    save_uncertainty_overlay(
                        image,
                        contours,
                        boxes,
                        uncertainties,
                        scores,
                        uncertainty_dir / f"{output_stem}.png",
                        f"Validation uncertainty | {source_name} | all predicted bubbles",
                    )
            if save_predictions and uncertainties is not None and boxes is not None:
                for index, value in enumerate(np.asarray(uncertainties).mean(axis=1)):
                    keep_uncertainty_candidate(
                        value, image, contours[index], boxes[index], uncertainties[index], scores[index]
                    )
    candidates = [entry[2] for entry in sorted(candidate_heap, key=lambda entry: entry[0], reverse=True)]
    return results, candidates


def average_f1(results: cd.data.LabelMatcherList) -> float:
    values = []
    for threshold in IOU_THRESHOLDS:
        results.iou_thresh = threshold
        values.append(float(results.avg_f1))
    return float(np.mean(values))


def validate_thresholds() -> tuple[float, float, float]:
    best = (-np.inf, model.score_thresh, model.nms_thresh)
    for score_threshold in conf.val_score_threshs:
        for nms_threshold in conf.val_nms_threshs:
            model.score_thresh, model.nms_thresh = score_threshold, nms_threshold
            results, _ = evaluate()
            value = average_f1(results)
            print(f"score_thresh={score_threshold}, nms_thresh={nms_threshold}, mean_f1={value:.4f}")
            best = max(best, (value, score_threshold, nms_threshold))
    best_f1, best_score, best_nms = best
    model.score_thresh, model.nms_thresh = best_score, best_nms
    return best_score, best_nms, best_f1


def show_uncertainty_results(candidates: list[tuple]) -> None:
    top_uncertainty_dir.mkdir(parents=True, exist_ok=True)
    for index, (mean_uncertainty, image, contour, box, uncertainty, score) in enumerate(candidates, start=1):
        x0, y0, x1, y1 = box.astype(int)
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(image, cmap="gray")
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color="tab:red", linewidth=1.2)
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.2))
        left, top, right, bottom = uncertainty
        style = {"color": "white", "fontsize": 5, "ha": "center", "va": "center", "bbox": {"facecolor": "black", "alpha": 0.55, "pad": 0.25}}
        axis.text((x0 + x1) / 2, y0, f"{top * 100:.0f}%", **style)
        axis.text((x0 + x1) / 2, y1, f"{bottom * 100:.0f}%", **style)
        axis.text(x0, (y0 + y1) / 2, f"{left * 100:.0f}%", rotation=90, **style)
        axis.text(x1, (y0 + y1) / 2, f"{right * 100:.0f}%", rotation=90, **style)
        axis.set(xlim=(max(0, x0 - 24), min(image.shape[1], x1 + 24)), ylim=(min(image.shape[0], y1 + 24), max(0, y0 - 24)), title=f"Score={score:.3f}, mean box uncertainty={mean_uncertainty:.4f}")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(top_uncertainty_dir / f"highest_uncertainty_{index:02d}.png", dpi=180)
        plt.close(figure)


def metrics_at_thresholds(results: cd.data.LabelMatcherList) -> dict:
    metrics = {}
    for threshold in IOU_THRESHOLDS:
        results.iou_thresh = threshold
        metrics[f"{threshold:.1f}"] = {"precision": float(results.precision), "recall": float(results.recall), "f1": float(results.avg_f1), "jaccard": float(results.avg_jaccard), "true_positives": int(results.true_positives), "false_positives": int(results.false_positives), "false_negatives": int(results.false_negatives)}
    return metrics


def plot_iou_metrics_and_counts(results: cd.data.LabelMatcherList) -> None:
    thresholds = np.arange(0.1, 1.0, 0.1)
    groups = (("Global-level metrics", ("precision", "recall", "f1_np", "jaccard_np", "fowlkes_mallows_np")), ("Sample-level metrics", ("avg_precision", "avg_recall", "avg_f1", "avg_jaccard", "avg_fowlkes_mallows")), ("Detection counts", ("true_positives", "false_positives", "false_negatives")))
    figure, axes = plt.subplots(1, 3, figsize=(24, 6))
    for axis, (title, keys) in zip(axes, groups):
        for key in keys:
            values = []
            for threshold in thresholds:
                results.iou_thresh = float(threshold)
                values.append(float(getattr(results, key)))
            axis.plot(thresholds, values, ".-", label=key)
        axis.set(xlabel="IoU threshold", title=title)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(figure_dir / "iou_metrics_and_counts.png", dpi=180)
    plt.close(figure)


train_losses = []
validation_losses = []
best_validation_f1 = -np.inf
best_model_path = output_dir / "best_model.pt"
for epoch in range(1, conf.epochs + 1):
    train_loss = train_epoch(epoch)
    validation_loss = validate_epoch()
    validation_results, _ = evaluate()
    validation_f1 = average_f1(validation_results)
    scheduler.step()
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    mlflow.log_metrics({"train_loss": train_loss, "validation_loss": validation_loss, "validation_mean_f1": validation_f1, "learning_rate": optimizer.param_groups[0]["lr"]}, step=epoch)
    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "validation_mean_f1": validation_f1}, best_model_path)
    if epoch % 10 == 0:
        show_results(epoch)


plot_loss_curves(train_losses, validation_losses)
checkpoint = torch.load(best_model_path, map_location=conf.device)
model.load_state_dict(checkpoint["model_state"])
score_threshold, nms_threshold, tuned_f1 = validate_thresholds()
torch.save({"epoch": checkpoint["epoch"], "model_state": model.state_dict(), "score_thresh": score_threshold, "nms_thresh": nms_threshold}, output_dir / "selected_model.pt")
final_results, uncertainty_candidates = evaluate(save_predictions=True)
final_f1 = average_f1(final_results)
show_uncertainty_results(uncertainty_candidates)
plot_iou_metrics_and_counts(final_results)
summary = {"best_epoch": checkpoint["epoch"], "best_epoch_mean_f1": checkpoint["validation_mean_f1"], "selected_score_thresh": score_threshold, "selected_nms_thresh": nms_threshold, "threshold_search_mean_f1": tuned_f1, "final_validation_mean_f1": final_f1, "metrics": metrics_at_thresholds(final_results)}
(output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
mlflow.log_metric("final_validation_mean_f1", final_f1)
mlflow.log_artifacts(str(output_dir), artifact_path="outputs")
print(json.dumps(summary, indent=2))
finish_mlflow_run()
