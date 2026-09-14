"""Compare HAICU training runs and their validation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def run_label(run_dir: Path, config: dict) -> str:
    return (
        f"{run_dir.name} (e={config.get('epochs', '?')}, "
        f"o={config.get('order', '?')}, s={config.get('samples', '?')})"
    )


def main() -> None:
    args = parse_args()
    runs = []
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        config = read_json(run_dir / "config.json", {})
        history = read_json(run_dir / "history.json", [])
        summary = read_json(run_dir / "validation_summary.json", {})
        if not history and (run_dir / "history.json").exists():
            raise ValueError(f"History in {run_dir} is not a list")
        runs.append({"path": run_dir, "config": config, "history": history, "summary": summary, "label": run_label(run_dir, config)})
    output_dir = (args.output_dir or runs[0]["path"] / "run_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    for run in runs:
        history = run["history"]
        if history:
            axis.plot(
                [item["epoch"] for item in history],
                [item["validation_mean_f1"] for item in history],
                label=run["label"],
            )
    axis.set(xlabel="Epoch", ylabel="Validation mean F1", title="Validation mean F1 by epoch")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_mean_f1_vs_epoch.png", dpi=180)
    plt.close(figure)

    metrics = ("precision", "recall", "f1", "jaccard")
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(runs))
    width = 0.8 / len(metrics)
    for metric_index, metric in enumerate(metrics):
        values = []
        for run in runs:
            metric_values = run["summary"].get("metrics", {}).get("0.5", {})
            values.append(float(metric_values.get(metric, np.nan)))
        axis.bar(positions + metric_index * width, values, width, label=metric)
    axis.set_xticks(positions + width * (len(metrics) - 1) / 2, [run["label"] for run in runs], rotation=30, ha="right")
    axis.set(ylabel="Validation metric", title="Validation metrics at IoU 0.5")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_metrics_comparison.png", dpi=180)
    plt.close(figure)

    calibration_metrics = (
        "score_ece",
        "score_mce",
        "ause",
        "uncertainty_auroc_fp",
        "uncertainty_spearman_error",
    )
    analysis_values = []
    for run in runs:
        analysis = read_json(run["path"] / "uncertainty_analysis" / "metrics.json", {})
        analysis_values.append(analysis.get("thresholds", {}).get("0.5", {}))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(calibration_metrics)
    for metric_index, metric in enumerate(calibration_metrics):
        values = [float(item.get(metric, np.nan)) for item in analysis_values]
        axis.bar(positions + metric_index * width, values, width, label=metric)
    axis.set_xticks(positions + width * (len(calibration_metrics) - 1) / 2, [run["label"] for run in runs], rotation=30, ha="right")
    axis.set_ylabel("Metric value")
    axis.set_title("Uncertainty and calibration comparison at IoU 0.5")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_metrics_comparison.png", dpi=180)
    plt.close(figure)

    payload = []
    for run in runs:
        payload.append({"run_dir": str(run["path"]), "config": run["config"], "validation_summary": run["summary"]})
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Wrote comparison figures to {output_dir}")


if __name__ == "__main__":
    main()
