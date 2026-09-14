"""Evaluate HAICU predictions, reliability, and box uncertainty calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import celldetection as cd
from bubble_coco_dataset import DEFAULT_DATA_ROOT, HAICUBubbleCPNDataset, parse_case_list
from uncertainty_analysis import make_prediction_records


IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run containing config.json and a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path; defaults to selected_model.pt or best_model.pt.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cases", help="Validation case IDs; defaults to split.json validation_cases.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--uncertainty-factor", type=float, default=None)
    parser.add_argument(
        "--uncertainty-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use uncertainty-aware NMS; default is false to avoid circular analysis.",
    )
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def config_value(config: dict, name: str, default):
    value = config.get(name, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def create_model(config: dict, args: argparse.Namespace) -> torch.nn.Module:
    model_name = config_value(config, "cpn", "CpnResNet50UNet")
    model_class = getattr(cd.models, model_name)
    model = model_class(
        in_channels=int(config_value(config, "in_channels", 1)),
        classes=int(config_value(config, "classes", 2)),
        order=int(config_value(config, "order", 6)),
        samples=int(config_value(config, "samples", 64)),
        refinement_iterations=int(config_value(config, "refinement_iterations", 3)),
        refinement_buckets=int(config_value(config, "refinement_buckets", 6)),
        contour_head_stride=int(config_value(config, "contour_head_stride", 2)),
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        uncertainty_head=True,
        uncertainty_nms=args.uncertainty_nms,
        uncertainty_factor=float(
            args.uncertainty_factor
            if args.uncertainty_factor is not None
            else config_value(config, "uncertainty_factor", 7.0)
        ),
        backbone_kwargs={
            "inputs_mean": float(config_value(config, "inputs_mean", 0.5)),
            "inputs_std": float(config_value(config, "inputs_std", 0.5)),
        },
    ).to(args.device)
    return model


def load_checkpoint(model: torch.nn.Module, path: Path, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()
    return checkpoint


def flatten_rows(rows: list[dict]) -> list[dict]:
    """Flatten list-valued fields for a CSV while retaining JSONL fidelity."""
    flat = []
    for row in rows:
        item = dict(row)
        for key in ("box", "box_uncertainty_raw", "box_uncertainty_sigma", "target_box", "box_error"):
            value = item.pop(key, None)
            if value is not None:
                for index, component in enumerate(value):
                    item[f"{key}_{index}"] = component
        flat.append(item)
    return flat


def write_rows(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "prediction_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=True) + "\n")
    flat_rows = flatten_rows(rows)
    fields = sorted({key for row in flat_rows for key in row})
    with (output_dir / "prediction_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def uncertainty_bins(rows: list[dict], threshold: float, bins: int) -> list[dict]:
    values = [row for row in rows if np.isfinite(row["mean_box_uncertainty"])]
    if not values:
        return []
    values.sort(key=lambda row: row["mean_box_uncertainty"])
    result = []
    for bin_index, group in enumerate(np.array_split(values, min(bins, len(values)))):
        group = list(group)
        correctness = np.asarray([row[f"correctness_iou_{threshold:.1f}".replace(".", "_")] for row in group])
        ious = np.asarray([row["best_iou"] for row in group if row["best_iou"] is not None], dtype=float)
        errors = np.asarray([row["rms_box_error"] for row in group if row["rms_box_error"] is not None], dtype=float)
        result.append(
            {
                "bin": bin_index,
                "count": len(group),
                "mean_uncertainty": float(np.mean([row["mean_box_uncertainty"] for row in group])),
                "min_uncertainty": float(group[0]["mean_box_uncertainty"]),
                "max_uncertainty": float(group[-1]["mean_box_uncertainty"]),
                "tp_rate": float(np.mean(correctness)),
                "fp_rate": float(1.0 - np.mean(correctness)),
                "mean_iou": float(np.mean(ious)) if len(ious) else math.nan,
                "mean_box_error": float(np.mean(errors)) if len(errors) else math.nan,
            }
        )
    return result


def risk_coverage(rows: list[dict], threshold: float, points: int = 101) -> dict:
    usable = [row for row in rows if np.isfinite(row["mean_box_uncertainty"])]
    if not usable:
        return {"coverage": [], "risk": [], "oracle_risk": [], "ause": math.nan}
    suffix = f"{threshold:.1f}".replace(".", "_")
    for row in usable:
        row["risk_error"] = 1.0 if not row[f"correctness_iou_{suffix}"] else 1.0 - float(row["best_iou"] or 0.0)
    uncertainty_order = sorted(usable, key=lambda row: row["mean_box_uncertainty"], reverse=True)
    oracle_order = sorted(usable, key=lambda row: row["risk_error"], reverse=True)
    coverage = np.linspace(0.0, 1.0, min(points, len(usable) + 1))
    risks = []
    oracle_risks = []
    for value in coverage:
        retained = int(round(value * len(usable)))
        if retained == 0:
            risks.append(0.0)
            oracle_risks.append(0.0)
        else:
            risks.append(float(np.mean([row["risk_error"] for row in uncertainty_order[-retained:]])))
            oracle_risks.append(float(np.mean([row["risk_error"] for row in oracle_order[-retained:]])))
    ause = float(np.trapezoid(np.maximum(0.0, np.asarray(risks) - np.asarray(oracle_risks)), coverage))
    return {"coverage": coverage.tolist(), "risk": risks, "oracle_risk": oracle_risks, "ause": ause}


def score_calibration(rows: list[dict], threshold: float, bins: int = 15, score_key: str = "score") -> tuple[list[dict], float, float]:
    suffix = f"{threshold:.1f}".replace(".", "_")
    usable = [row for row in rows if np.isfinite(row[score_key])]
    if not usable:
        return [], math.nan, math.nan
    output = []
    gaps = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        group = [row for row in usable if edges[index] <= row[score_key] < edges[index + 1] or (index == bins - 1 and row[score_key] == 1.0)]
        if group:
            confidence = float(np.mean([row[score_key] for row in group]))
            accuracy = float(np.mean([row[f"correctness_iou_{suffix}"] for row in group]))
            gap = abs(confidence - accuracy)
            gaps.append((len(group), gap))
        else:
            confidence = accuracy = gap = math.nan
        output.append({"bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": len(group), "mean_score": confidence, "accuracy": accuracy, "gap": gap})
    total = sum(count for count, _ in gaps)
    ece = float(sum(count * gap for count, gap in gaps) / total) if total else math.nan
    mce = float(max(gap for _, gap in gaps)) if gaps else math.nan
    return output, ece, mce


def temperature_crossval(rows: list[dict], threshold: float) -> dict:
    """Fit one score temperature per case-held-out fold and report ECE."""
    cases = {}
    for row in rows:
        cases.setdefault(str(row.get("case_id", "unknown")), []).append(row)
    if len(cases) < 2:
        return {"available": False, "reason": "at least two cases are required"}
    suffix = f"{threshold:.1f}".replace(".", "_")
    temperatures = []
    heldout_ece = []
    for heldout, test_rows in cases.items():
        fit_rows = [row for case, values in cases.items() if case != heldout for row in values]
        scores = np.asarray([row["score"] for row in fit_rows], dtype=float)
        labels = np.asarray([row[f"correctness_iou_{suffix}"] for row in fit_rows], dtype=float)
        finite = np.isfinite(scores)
        scores = np.clip(scores[finite], 1e-6, 1 - 1e-6)
        labels = labels[finite]
        if len(scores) == 0 or len(np.unique(labels)) < 2:
            continue
        logits = np.log(scores / (1.0 - scores))
        def nll(log_temperature):
            scaled = logits / np.exp(log_temperature)
            return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))
        result = optimize.minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        temperature = float(np.exp(result.x))
        temperatures.append(temperature)
        for row in test_rows:
            score = float(np.clip(row["score"], 1e-6, 1 - 1e-6))
            row["score_temperature_scaled"] = float(1.0 / (1.0 + np.exp(-np.log(score / (1.0 - score)) / temperature)))
        _, ece, _ = score_calibration(test_rows, threshold, 15, score_key="score_temperature_scaled")
        if np.isfinite(ece):
            heldout_ece.append(ece)
    return {"available": bool(temperatures), "temperatures": temperatures, "mean_temperature": float(np.mean(temperatures)) if temperatures else math.nan, "heldout_ece": heldout_ece, "mean_heldout_ece": float(np.mean(heldout_ece)) if heldout_ece else math.nan}


def sigma_crossval(rows: list[dict], threshold: float) -> dict:
    """Fit a scalar sigma multiplier on other cases and score held-out coverage."""
    cases = {}
    suffix = f"{threshold:.1f}".replace(".", "_")
    for row in rows:
        cases.setdefault(str(row.get("case_id", "unknown")), []).append(row)
    factors, coverages = [], []
    for heldout, test_rows in cases.items():
        fit_rows = [row for case, values in cases.items() if case != heldout for row in values]
        fit_errors, fit_sigmas = [], []
        for row in fit_rows:
            if row[f"correctness_iou_{suffix}"] and row["box_error"] is not None:
                fit_errors.extend(np.abs(np.asarray(row["box_error"], dtype=float)))
                fit_sigmas.extend(np.asarray(row["box_uncertainty_sigma"], dtype=float))
        valid = np.isfinite(fit_errors) & np.isfinite(fit_sigmas) & (np.asarray(fit_sigmas) > 0)
        if not valid.any():
            continue
        factor = float(np.sqrt(np.mean(np.asarray(fit_errors)[valid] ** 2) / np.mean(np.asarray(fit_sigmas)[valid] ** 2)))
        factors.append(factor)
        covered = []
        for row in test_rows:
            if row[f"correctness_iou_{suffix}"] and row["box_error"] is not None:
                sigma = np.asarray(row["box_uncertainty_sigma"], dtype=float) * factor
                error = np.abs(np.asarray(row["box_error"], dtype=float))
                covered.extend((error <= stats.norm.ppf(0.84) * sigma).tolist())
        if covered:
            coverages.append(float(np.mean(covered)))
    return {"available": bool(factors), "factors": factors, "mean_factor": float(np.mean(factors)) if factors else math.nan, "heldout_coverage_68": coverages, "mean_heldout_coverage_68": float(np.mean(coverages)) if coverages else math.nan}


def interval_calibration(rows: list[dict], threshold: float, edge: int, bins: int = 100) -> dict:
    suffix = f"{threshold:.1f}".replace(".", "_")
    matched = [
        row for row in rows
        if row[f"correctness_iou_{suffix}"] and row["target_box"] is not None
        and np.isfinite(row["box_uncertainty_sigma"][edge])
        and row["box_uncertainty_sigma"][edge] > 0
    ]
    expected = np.linspace(0.01, 0.99, bins)
    observed = []
    for probability in expected:
        z = stats.norm.ppf(0.5 + probability / 2.0)
        observed.append(float(np.mean([
            abs(row["box_error"][edge]) <= z * row["box_uncertainty_sigma"][edge]
            for row in matched
        ])) if matched else math.nan)
    return {"expected": expected.tolist(), "observed": observed, "count": len(matched)}


def mean_interval_calibration(rows: list[dict], threshold: float, bins: int = 100) -> dict:
    """Calibration of mean sigma against RMS error across four box edges."""
    suffix = f"{threshold:.1f}".replace(".", "_")
    matched = [
        row for row in rows
        if row[f"correctness_iou_{suffix}"] and row["box_error"] is not None
        and np.isfinite(row["mean_box_sigma"]) and row["mean_box_sigma"] > 0
    ]
    expected = np.linspace(0.01, 0.99, bins)
    observed = []
    for probability in expected:
        z = stats.norm.ppf(0.5 + probability / 2.0)
        observed.append(float(np.mean([
            np.sqrt(np.mean(np.square(row["box_error"]))) <= z * row["mean_box_sigma"]
            for row in matched
        ])) if matched else math.nan)
    return {"expected": expected.tolist(), "observed": observed, "count": len(matched)}


def uncertainty_scatter(rows: list[dict], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    suffix = f"{threshold:.1f}".replace(".", "_")
    matched = [row for row in rows if row[f"correctness_iou_{suffix}"] and row["rms_box_error"] is not None]
    if not matched:
        return np.empty(0), np.empty(0)
    return (
        np.asarray([row["mean_box_sigma"] for row in matched], dtype=float),
        np.asarray([row["rms_box_error"] for row in matched], dtype=float),
    )


def combined_risk_coverage(rows: list[dict], threshold: float, points: int = 101) -> dict:
    """Risk--coverage using a normalized score/uncertainty ranking."""
    usable = [row for row in rows if np.isfinite(row["score"]) and np.isfinite(row["mean_box_uncertainty"])]
    if not usable:
        return {"coverage": [], "risk": [], "ause": math.nan}
    suffix = f"{threshold:.1f}".replace(".", "_")
    for row in usable:
        row["risk_error"] = 1.0 if not row[f"correctness_iou_{suffix}"] else 1.0 - float(row["best_iou"] or 0.0)
    scores = np.asarray([row["score"] for row in usable])
    uncertainties = np.asarray([row["mean_box_uncertainty"] for row in usable])
    score_scale = np.ptp(scores) or 1.0
    uncertainty_scale = np.ptp(uncertainties) or 1.0
    for row, score, uncertainty in zip(usable, scores, uncertainties):
        row["combined_risk"] = float((1.0 - score) / score_scale + uncertainty / uncertainty_scale)
    ordered = sorted(usable, key=lambda row: row["combined_risk"], reverse=True)
    oracle = sorted(usable, key=lambda row: row["risk_error"], reverse=True)
    coverage = np.linspace(0.0, 1.0, min(points, len(usable) + 1))
    risks = []
    oracle_risks = []
    for value in coverage:
        retained = int(round(value * len(usable)))
        if retained == 0:
            risks.append(0.0); oracle_risks.append(0.0)
        else:
            risks.append(float(np.mean([row["risk_error"] for row in ordered[-retained:]])))
            oracle_risks.append(float(np.mean([row["risk_error"] for row in oracle[-retained:]])))
    ause = float(np.trapezoid(np.maximum(0.0, np.asarray(risks) - np.asarray(oracle_risks)), coverage))
    return {"coverage": coverage.tolist(), "risk": risks, "oracle_risk": oracle_risks, "ause": ause}


def save_uncertainty_diagnostics(rows: list[dict], output_dir: Path, threshold: float) -> None:
    suffix = f"{threshold:.1f}".replace(".", "_")
    tp = [row["mean_box_uncertainty"] for row in rows if row[f"correctness_iou_{suffix}"]]
    fp = [row["mean_box_uncertainty"] for row in rows if not row[f"correctness_iou_{suffix}"]]
    figure, axis = plt.subplots(figsize=(6, 4))
    if tp:
        axis.hist(tp, bins=30, alpha=0.55, density=True, label="TP")
    if fp:
        axis.hist(fp, bins=30, alpha=0.55, density=True, label="FP")
    axis.set(xlabel="Mean box uncertainty", ylabel="Density", title=f"TP/FP uncertainty (IoU={threshold:.1f})")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"tp_fp_uncertainty_iou_{suffix}.png", dpi=180)
    plt.close(figure)

    labels = np.asarray([int(not row[f"correctness_iou_{suffix}"]) for row in rows])
    scores = np.asarray([row["mean_box_uncertainty"] for row in rows], dtype=float)
    valid = np.isfinite(scores)
    if valid.sum() and len(np.unique(labels[valid])) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels[valid], scores[valid])
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.plot(false_positive_rate, true_positive_rate, label=f"AUROC={roc_auc_score(labels[valid], scores[valid]):.3f}")
        axis.plot([0, 1], [0, 1], "--", color="0.5")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"Uncertainty FP ROC (IoU={threshold:.1f})")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"uncertainty_fp_roc_iou_{suffix}.png", dpi=180)
        plt.close(figure)

        score_values = np.asarray([1.0 - row["score"] for row in rows], dtype=float)
        combined_values = score_values + np.nan_to_num(scores, nan=0.0)
        figure, axis = plt.subplots(figsize=(5, 5))
        for name, values in (("mean uncertainty", scores), ("1 - score", score_values), ("score + uncertainty", combined_values)):
            finite = valid & np.isfinite(values)
            if finite.sum() and len(np.unique(labels[finite])) == 2:
                false_positive_rate, true_positive_rate, _ = roc_curve(labels[finite], values[finite])
                area = roc_auc_score(labels[finite], values[finite])
                axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUROC={area:.3f})")
        axis.plot([0, 1], [0, 1], "--", color="0.5")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"FP ranking comparison (IoU={threshold:.1f})")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"fp_ranking_comparison_iou_{suffix}.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    saturation = np.asarray([row["box_uncertainty_raw"] for row in rows], dtype=float).reshape(-1)
    saturation = saturation[np.isfinite(saturation)]
    if len(saturation):
        axis.hist(saturation, bins=30, range=(0, 1), alpha=0.8)
    axis.axvline(0.95, linestyle="--", color="tab:red", label="5% below sigmoid ceiling")
    axis.set(xlabel="Raw box uncertainty", ylabel="Count", title="Uncertainty-head saturation")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"uncertainty_saturation_iou_{suffix}.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=False)
    for edge, axis in enumerate(axes.flat):
        values = []
        for row in rows:
            suffix_edge = f"{threshold:.1f}".replace(".", "_")
            if row[f"correctness_iou_{suffix_edge}"] and row["box_uncertainty_sigma"][edge] > 0:
                values.append(row["box_error"][edge] / row["box_uncertainty_sigma"][edge])
        if values:
            axis.hist(values, bins=30, density=True, alpha=0.7)
            x = np.linspace(-4, 4, 200)
            axis.plot(x, stats.norm.pdf(x), "--", label="N(0,1)")
        axis.set_title(("left", "top", "right", "bottom")[edge])
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"Standardised box errors (IoU={threshold:.1f})")
    figure.tight_layout()
    figure.savefig(output_dir / f"zscore_histogram_iou_{suffix}.png", dpi=180)
    plt.close(figure)

    sigma, error = uncertainty_scatter(rows, threshold)
    if len(sigma):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(sigma, error, s=10, alpha=0.45)
        if len(sigma) > 1:
            correlation = stats.spearmanr(sigma, error).statistic
            axis.set_title(f"Residuals versus mean sigma (Spearman={correlation:.3f})")
        else:
            axis.set_title("Residuals versus mean sigma")
        axis.set(xlabel="Mean box sigma (pixels)", ylabel="RMS box error (pixels)")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / f"residuals_vs_mean_sigma_iou_{suffix}.png", dpi=180)
        plt.close(figure)


def save_plots(rows: list[dict], output_dir: Path, threshold: float, bins: int) -> dict:
    summaries = uncertainty_bins(rows, threshold, bins)
    with (output_dir / f"uncertainty_bins_iou_{threshold:.1f}.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2, allow_nan=True)
    if summaries:
        x = [item["mean_uncertainty"] for item in summaries]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(x, [item["tp_rate"] for item in summaries], ".-", label="TP rate")
        axes[0].plot(x, [item["fp_rate"] for item in summaries], ".-", label="FP rate")
        axes[0].set(xlabel="Mean box uncertainty", ylabel="Rate", title="Uncertainty reliability")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(x, [item["mean_iou"] for item in summaries], ".-", label="Mean IoU")
        axes[1].plot(x, [item["mean_box_error"] for item in summaries], ".-", label="Box error")
        axes[1].set(xlabel="Mean box uncertainty", ylabel="Error / IoU", title="Uncertainty versus quality")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / f"uncertainty_bins_iou_{threshold:.1f}.png", dpi=180)
        plt.close(figure)

    risk = risk_coverage(rows, threshold)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(risk["coverage"], risk["risk"], label="Uncertainty ordering")
    axis.plot(risk["coverage"], risk["oracle_risk"], "--", label="Oracle ordering")
    axis.set(xlabel="Coverage (lowest uncertainty retained)", ylabel="Risk", title=f"Risk--coverage (AUSE={risk['ause']:.4f})")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"risk_coverage_iou_{threshold:.1f}.png", dpi=180)
    plt.close(figure)

    score_bins, ece, mce = score_calibration(rows, threshold, 15)
    temperature = temperature_crossval(rows, threshold)
    sigma_recalibration = sigma_crossval(rows, threshold)
    if score_bins:
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.plot([0, 1], [0, 1], "--", label="Ideal")
        axis.plot([item["mean_score"] for item in score_bins], [item["accuracy"] for item in score_bins], ".-", label="Model")
        scaled_bins, _, _ = score_calibration(rows, threshold, 15, score_key="score_temperature_scaled") if temperature.get("available") else ([], math.nan, math.nan)
        if scaled_bins:
            axis.plot([item["mean_score"] for item in scaled_bins], [item["accuracy"] for item in scaled_bins], ".-", label="Temperature scaled")
        axis.set(xlabel="Mean detection score", ylabel="Empirical correctness", title=f"Score calibration (ECE={ece:.4f}, MCE={mce:.4f})")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"score_reliability_iou_{threshold:.1f}.png", dpi=180)
        plt.close(figure)

    calibration = {"mean_sigma": mean_interval_calibration(rows, threshold)}
    edge_names = ("left", "top", "right", "bottom")
    for edge in range(4):
        calibration[str(edge)] = interval_calibration(rows, threshold, edge)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", label="Ideal")
    for key, values in calibration.items():
        label = "mean sigma" if key == "mean_sigma" else edge_names[int(key)]
        axis.plot(values["expected"], values["observed"], label=label)
    axis.set(xlabel="Expected interval coverage", ylabel="Observed coverage", title=f"Box uncertainty calibration (IoU={threshold:.1f})")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"box_calibration_iou_{threshold:.1f}.png", dpi=180)
    plt.close(figure)
    save_uncertainty_diagnostics(rows, output_dir, threshold)
    combined = combined_risk_coverage(rows, threshold)
    return {"bins": score_bins, "score_ece": ece, "score_mce": mce, "temperature_scaling": temperature, "sigma_recalibration": sigma_recalibration, "risk_coverage": risk, "combined_risk_coverage": combined, "interval_calibration": calibration}


def optional_toolbox_metrics(rows: list[dict], threshold: float) -> dict:
    """Compute toolbox metrics when the optional package is installed."""
    try:
        import uncertainty_toolbox as uct
    except ImportError:
        return {"available": False, "reason": "uncertainty-toolbox is not installed"}
    suffix = f"{threshold:.1f}".replace(".", "_")
    result = {"available": True, "edges": {}}
    for edge, name in enumerate(("left", "top", "right", "bottom")):
        matched = [
            row for row in rows
            if row[f"correctness_iou_{suffix}"] and row["target_box"] is not None
            and row["box_uncertainty_sigma"][edge] > 0
        ]
        if not matched:
            result["edges"][name] = {"count": 0}
            continue
        y_pred = np.asarray([row["box"][edge] for row in matched], dtype=float)
        y_true = np.asarray([row["target_box"][edge] for row in matched], dtype=float)
        y_std = np.asarray([row["box_uncertainty_sigma"][edge] for row in matched], dtype=float)
        result["edges"][name] = {
            "count": len(matched),
            "metrics": {
                "mace": float(uct.mean_absolute_calibration_error(y_pred, y_std, y_true)),
                "rmsce": float(uct.root_mean_squared_calibration_error(y_pred, y_std, y_true)),
                "miscalibration_area": float(uct.miscalibration_area(y_pred, y_std, y_true)),
                "sharpness": float(uct.sharpness(y_std)),
                "nll": float(uct.nll_gaussian(y_pred, y_std, y_true)),
                "crps": float(uct.crps_gaussian(y_pred, y_std, y_true)),
                "interval": float(uct.interval_score(y_pred, y_std, y_true)),
            },
        }
    return result


def case_bootstrap(rows: list[dict], threshold: float, repetitions: int = 500, seed: int = 42) -> dict:
    """Bootstrap validation metrics by case, preserving frame/view correlation."""
    cases = {}
    for row in rows:
        cases.setdefault(str(row.get("case_id", "unknown")), []).append(row)
    if len(cases) < 2:
        return {"count": 0, "reason": "at least two case IDs are required"}
    case_rows = list(cases.values())
    generator = np.random.default_rng(seed)
    suffix = f"{threshold:.1f}".replace(".", "_")

    def calculate(sample):
        labels = np.asarray([int(not row[f"correctness_iou_{suffix}"]) for row in sample])
        uncertainty = np.asarray([row["mean_box_uncertainty"] for row in sample], dtype=float)
        valid = np.isfinite(uncertainty)
        auroc = roc_auc_score(labels[valid], uncertainty[valid]) if valid.sum() and len(np.unique(labels[valid])) == 2 else math.nan
        tp_rows = [row for row in sample if row[f"correctness_iou_{suffix}"] and row["best_iou"] is not None]
        spearman = stats.spearmanr([row["mean_box_uncertainty"] for row in tp_rows], [1.0 - row["best_iou"] for row in tp_rows]).statistic if len(tp_rows) > 1 else math.nan
        _, ece, _ = score_calibration(sample, threshold, 15)
        ause = risk_coverage(sample, threshold)["ause"]
        return {"uncertainty_auroc_fp": auroc, "uncertainty_spearman_error": spearman, "score_ece": ece, "ause": ause}

    values = {name: [] for name in ("uncertainty_auroc_fp", "uncertainty_spearman_error", "score_ece", "ause")}
    for _ in range(repetitions):
        sample = [row for index in generator.integers(0, len(case_rows), size=len(case_rows)) for row in case_rows[index]]
        result = calculate(sample)
        for name, value in result.items():
            if np.isfinite(value):
                values[name].append(float(value))
    intervals = {}
    for name, value in values.items():
        intervals[name] = {
            "estimate": float(calculate(rows)[name]) if np.isfinite(calculate(rows)[name]) else math.nan,
            "lower": float(np.percentile(value, 2.5)) if value else math.nan,
            "upper": float(np.percentile(value, 97.5)) if value else math.nan,
            "n": len(value),
        }
    return {"case_count": len(case_rows), "repetitions": repetitions, "intervals": intervals}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "config.json")
    split = load_json(run_dir / "split.json") if (run_dir / "split.json").exists() else {}
    case_ids = parse_case_list(args.cases) if args.cases else split.get("validation_cases", [5, 10, 11])
    data_root = args.data_root or Path(
        config_value(config, "data_root", config_value(config, "directory", DEFAULT_DATA_ROOT))
    )
    output_dir = (args.output_dir or run_dir / "uncertainty_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = next((run_dir / name for name in ("selected_model.pt", "best_model.pt" ) if (run_dir / name).exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No selected_model.pt or best_model.pt in {run_dir}")

    model = create_model(config, args)
    checkpoint = load_checkpoint(model, checkpoint_path, args.device)
    factor = float(args.uncertainty_factor if args.uncertainty_factor is not None else config_value(config, "uncertainty_factor", 7.0))
    dataset = HAICUBubbleCPNDataset(
        data_root,
        case_ids,
        samples=int(config_value(config, "samples", 64)),
        order=int(config_value(config, "order", 6)),
        min_visible_area_fraction=float(config_value(config, "min_visible_area_fraction", 0.75)),
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=cd.universal_dict_collate_fn)
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="Collecting predictions")):
            device_batch = cd.to_device(batch, args.device)
            outputs = cd.asnumpy(model(device_batch["inputs"]))
            numpy_batch = cd.asnumpy(batch)
            target = cd.data.channels_first2channels_last(numpy_batch["targets"][0])
            contours = outputs["contours"][0]
            boxes = outputs.get("boxes", [None])[0]
            scores = outputs.get("scores", [None])[0]
            uncertainties = outputs.get("box_uncertainties", [None])[0]
            sample_rows = make_prediction_records(
                contours, boxes, scores, uncertainties, target,
                iou_thresholds=IOU_THRESHOLDS, uncertainty_factor=factor,
            )
            record = dataset.records[index]
            for row in sample_rows:
                row.update({"case_id": record["case_id"], "image_id": record["image_id"], "file_name": record["file_name"], "sample_index": index})
            rows.extend(sample_rows)

    write_rows(rows, output_dir)
    summaries = {"run_dir": str(run_dir), "checkpoint": str(checkpoint_path), "cases": case_ids, "score_thresh": args.score_thresh, "nms_thresh": args.nms_thresh, "uncertainty_nms": args.uncertainty_nms, "uncertainty_factor": factor, "num_predictions": len(rows), "thresholds": {}}
    for threshold in IOU_THRESHOLDS:
        analysis = save_plots(rows, output_dir, threshold, args.bins)
        suffix = f"{threshold:.1f}".replace(".", "_")
        usable = [row for row in rows if row[f"correctness_iou_{suffix}"] or row["best_iou"] is not None]
        tp_rows = [row for row in rows if row[f"correctness_iou_{suffix}"] and row["best_iou"] is not None]
        fp_labels = np.asarray([int(not row[f"correctness_iou_{suffix}"]) for row in rows])
        unc_values = np.asarray([row["mean_box_uncertainty"] for row in rows], dtype=float)
        valid = np.isfinite(unc_values)
        auroc = float(roc_auc_score(fp_labels[valid], unc_values[valid])) if valid.sum() and len(np.unique(fp_labels[valid])) == 2 else math.nan
        spearman = float(stats.spearmanr([row["mean_box_uncertainty"] for row in tp_rows], [1.0 - row["best_iou"] for row in tp_rows]).statistic) if len(tp_rows) >= 2 else math.nan
        summaries["thresholds"][f"{threshold:.1f}"] = {
            "num_predictions": len(usable),
            "num_tp": sum(row[f"correctness_iou_{suffix}"] for row in rows),
            "num_fp": sum(not row[f"correctness_iou_{suffix}"] for row in rows),
            "uncertainty_auroc_fp": auroc,
            "uncertainty_spearman_error": spearman,
            "score_ece": analysis["score_ece"],
            "score_mce": analysis["score_mce"],
            "ause": analysis["risk_coverage"]["ause"],
            "combined_ause": analysis["combined_risk_coverage"]["ause"],
            "temperature_scaling": analysis["temperature_scaling"],
            "sigma_recalibration": analysis["sigma_recalibration"],
            "case_bootstrap": case_bootstrap(rows, threshold),
            "toolbox": optional_toolbox_metrics(rows, threshold),
        }
    (output_dir / "metrics.json").write_text(json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
