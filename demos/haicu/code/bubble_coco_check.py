"""Audit HAICU COCO polygons and save image-overlay examples before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import celldetection as cd
from bubble_coco_dataset import (
    DEFAULT_DATA_ROOT,
    HAICUBubbleCPNDataset,
    parse_case_list,
    select_visible_contours,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cases", default="1", help="Comma-separated case IDs")
    parser.add_argument("--examples-per-case", type=int, default=4)
    parser.add_argument(
        "--check-cpn-targets",
        action="store_true",
        help="Also visualise the clipped, overlap-aware CPN targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "coco_check",
    )
    return parser.parse_args()


def draw_overlay(image: np.ndarray, contours: list[np.ndarray], output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 12))
    axis.imshow(image, cmap="gray")
    for contour in contours:
        closed = np.vstack((contour, contour[:1]))
        axis.plot(closed[:, 0], closed[:, 1], linewidth=0.7)
    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def cpn_target_contours(targets: np.ndarray) -> list[np.ndarray]:
    contours = cd.data.labels2contours(targets)
    if isinstance(contours, dict):
        contours = contours.values()
    return [np.squeeze(contour, axis=1) if contour.ndim == 3 else contour for contour in contours]


def check_case(
    dataset: HAICUBubbleCPNDataset,
    case_id: int,
    examples: int,
    output_dir: Path,
    check_cpn_targets: bool,
) -> None:
    indices = dataset.case_indices[case_id]
    annotation_count = 0
    boundary_annotation_count = 0
    discarded_annotation_count = 0
    cameras = set()
    for offset, index in enumerate(indices):
        record, image, contours = dataset.raw_sample(index)
        annotation_count += len(contours)
        cameras.add(Path(record["file_name"]).parent.name)
        for contour in contours:
            crosses_image_boundary = (
                np.any(contour[:, 0] < 0)
                or np.any(contour[:, 0] >= image.shape[1])
                or np.any(contour[:, 1] < 0)
                or np.any(contour[:, 1] >= image.shape[0])
            )
            boundary_annotation_count += int(crosses_image_boundary)
        kept_contours, _ = select_visible_contours(contours, image.shape)
        discarded_annotation_count += len(contours) - len(kept_contours)
        if offset < examples:
            draw_overlay(
                image,
                contours,
                output_dir / f"Train_{case_id}_{Path(record['file_name']).parent.name}_{index:04d}.png",
                f"{record['file_name']} | {len(contours)} bubbles",
            )
            if check_cpn_targets:
                sample = dataset[index]
                target_contours = cpn_target_contours(sample["targets"])
                draw_overlay(
                    image,
                    target_contours,
                    output_dir / f"Train_{case_id}_{Path(record['file_name']).parent.name}_{index:04d}_cpn_targets.png",
                    f"CPN targets: {record['file_name']} | {len(target_contours)} bubbles",
                )
                if offset == 0:
                    print(
                        "CPN sample shapes: "
                        f"inputs={sample['inputs'].shape}, targets={sample['targets'].shape}, "
                        f"fourier={sample['fourier'][0].shape}, locations={sample['locations'][0].shape}"
                    )
    print(
        f"Train_{case_id}: {len(indices)} images, {annotation_count} annotations, "
        f"boundary-crossing annotations={boundary_annotation_count}, "
        f"discarded below 75% visible area={discarded_annotation_count}, cameras={sorted(cameras)}"
    )


def main() -> None:
    args = parse_args()
    case_ids = parse_case_list(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HAICUBubbleCPNDataset(args.data_root, case_ids)
    for case_id in case_ids:
        check_case(dataset, case_id, args.examples_per_case, args.output_dir, args.check_cpn_targets)


if __name__ == "__main__":
    main()
