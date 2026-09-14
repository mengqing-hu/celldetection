"""Utilities for per-instance HAICU detection and uncertainty analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np

import celldetection as cd


def label_bbox(labels: np.ndarray, label: int) -> np.ndarray | None:
    """Return an ``xyxy`` bounding box for one label in an HWC label array."""
    mask = (labels == label).any(axis=-1)
    rows, columns = np.where(mask)
    if len(rows) == 0:
        return None
    return np.asarray(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.float32,
    )


def contour_bbox(contour: np.ndarray) -> np.ndarray:
    """Return an ``xyxy`` bounding box for a contour."""
    contour = np.asarray(contour)
    return np.asarray(
        [contour[:, 0].min(), contour[:, 1].min(), contour[:, 0].max(), contour[:, 1].max()],
        dtype=np.float32,
    )


def _best_match(matcher: cd.data.LabelMatcher, input_label: int) -> tuple[int | None, float | None]:
    """Return the highest-IoU target label and IoU for one input label."""
    if matcher.matches is None or len(matcher.matches) == 0:
        return None, None
    indices = np.flatnonzero(matcher.matches[:, 0] == input_label)
    if len(indices) == 0:
        return None, None
    index = indices[np.argmax(matcher.ious[indices])]
    return int(matcher.matches[index, 1]), float(matcher.ious[index])


def make_prediction_records(
    contours: Iterable[np.ndarray],
    boxes: np.ndarray | None,
    scores: np.ndarray | None,
    uncertainties: np.ndarray | None,
    target_labels: np.ndarray,
    iou_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    uncertainty_factor: float = 7.0,
) -> list[dict]:
    """Match predictions to target instances and return JSON-serialisable rows.

    Matching uses the same rasterised amodal labels and greedy IoU filtering as
    :class:`celldetection.data.LabelMatcher`.  The raw best match is retained
    for diagnostics, while ``tp_iou_*`` fields use the thresholded matcher.
    """
    contours = list(contours)
    target_labels = np.asarray(target_labels)
    prediction_labels = cd.data.contours2labels(contours, target_labels.shape[:2])
    matcher = cd.data.LabelMatcher(prediction_labels, target_labels)
    thresholds = tuple(float(value) for value in iou_thresholds)

    if boxes is None:
        boxes = np.asarray([contour_bbox(contour) for contour in contours], dtype=np.float32)
    else:
        boxes = np.asarray(boxes)
    if scores is None:
        scores = np.full(len(contours), np.nan, dtype=np.float32)
    else:
        scores = np.asarray(scores)
    if uncertainties is None:
        uncertainties = np.full((len(contours), 4), np.nan, dtype=np.float32)
    else:
        uncertainties = np.asarray(uncertainties)

    rows = []
    for index in range(len(contours)):
        input_label = index + 1
        target_label, best_iou = _best_match(matcher, input_label)
        prediction_box = np.asarray(boxes[index], dtype=np.float32)
        height, width = target_labels.shape[:2]
        box_width = max(0.0, float(prediction_box[2] - prediction_box[0]))
        box_height = max(0.0, float(prediction_box[3] - prediction_box[1]))
        row = {
            "prediction_index": int(index),
            "score": float(scores[index]),
            "box": prediction_box.tolist(),
            "box_uncertainty_raw": np.asarray(uncertainties[index], dtype=np.float32).tolist(),
            "box_uncertainty_sigma": (np.asarray(uncertainties[index], dtype=np.float32) * uncertainty_factor).tolist(),
            "mean_box_uncertainty": float(np.nanmean(uncertainties[index])),
            "mean_box_sigma": float(np.nanmean(uncertainties[index]) * uncertainty_factor),
            "best_target_label": target_label,
            "best_iou": best_iou,
            "box_area": box_width * box_height,
            "box_touches_border": bool(
                prediction_box[0] <= 0
                or prediction_box[1] <= 0
                or prediction_box[2] >= width - 1
                or prediction_box[3] >= height - 1
            ),
        }
        target_box = label_bbox(target_labels, target_label) if target_label is not None else None
        row["target_box"] = target_box.tolist() if target_box is not None else None
        if target_box is not None:
            row["box_error"] = (prediction_box - target_box).tolist()
            row["rms_box_error"] = float(np.sqrt(np.mean(np.square(prediction_box - target_box))))
        else:
            row["box_error"] = None
            row["rms_box_error"] = None

        for threshold in thresholds:
            matcher.iou_thresh = threshold
            true_positive_labels = matcher.true_positive_labels
            is_tp = input_label in true_positive_labels
            suffix = f"{threshold:.1f}".replace(".", "_")
            row[f"tp_iou_{suffix}"] = bool(is_tp)
            row[f"correctness_iou_{suffix}"] = int(is_tp)
            row[f"error_iou_{suffix}"] = float(not is_tp)
        rows.append(row)
    return rows
