"""Lazy COCO dataset for the initial HAICU bubble CPN experiment.

The implementation intentionally keeps only one case's annotations in memory.
The case-aware batch sampler keeps samples from one case together, so images can
be shuffled without parsing every large COCO file for every batch.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
from torch.utils.data import Dataset, Sampler

import celldetection as cd


DEFAULT_DATA_ROOT = Path(
    "/bigdata/haicu/starke88/data/from_hendrik_hessenkemper_fwdc/"
    "bubble_multicamera_for_paper"
)
MIN_VISIBLE_AREA_FRACTION = 0.75


def parse_case_list(value: str) -> list[int]:
    """Parse a comma-separated case list such as ``1,2,5``."""
    return [int(item) for item in value.split(",") if item.strip()]


def _case_annotation_path(data_root: Path, case_id: int) -> Path:
    return data_root / f"Train_{case_id}" / "coco_annotation_seg.json"


def load_case_coco(data_root: Path, case_id: int) -> dict:
    """Load the complete segmentation COCO document for one case."""
    annotation_path = _case_annotation_path(data_root, case_id)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO segmentation file: {annotation_path}")
    with annotation_path.open() as annotation_file:
        return json.load(annotation_file)


def list_case_records(data_root: Path, case_id: int) -> list[dict]:
    """Return image metadata only; annotation polygons are loaded lazily later."""
    coco = load_case_coco(data_root, case_id)
    records = []
    for image in coco["images"]:
        image_path = data_root / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image path does not exist: {image_path}")
        records.append(
            {
                "case_id": case_id,
                "image_id": image["id"],
                "file_name": image["file_name"],
                "height": image["height"],
                "width": image["width"],
            }
        )
    return records


def annotation_contours(annotations: list[dict]) -> list[np.ndarray]:
    """Convert one-polygon COCO segmentations to ``(x, y)`` contour arrays."""
    contours = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])
        if len(segmentation) != 1:
            raise ValueError(
                "This initial loader expects exactly one polygon per bubble. "
                f"Annotation {annotation.get('id')} has {len(segmentation)} polygons."
            )
        polygon = np.asarray(segmentation[0], dtype=np.float32)
        if polygon.size < 6 or polygon.size % 2:
            raise ValueError(f"Invalid polygon in annotation {annotation.get('id')}")
        contours.append(polygon.reshape(-1, 2))
    return contours


def polygon_area(contour: np.ndarray) -> float:
    if len(contour) < 3:
        return 0.0
    x_coordinates = contour[:, 0]
    y_coordinates = contour[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_coordinates, np.roll(y_coordinates, -1))
            - np.dot(y_coordinates, np.roll(x_coordinates, -1))
        )
    )


def clip_polygon_to_image(contour: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Clip a polygon to the continuous image rectangle ``[0, width] x [0, height]``."""
    height, width = image_shape
    vertices = contour.astype(np.float64, copy=True)

    def clip_edge(points, inside, intersection):
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        clipped = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                clipped.append(intersection(previous, current))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        return np.asarray(clipped, dtype=np.float64)

    def intersection_x(first, second, boundary):
        fraction = (boundary - first[0]) / (second[0] - first[0])
        return np.array((boundary, first[1] + fraction * (second[1] - first[1])))

    def intersection_y(first, second, boundary):
        fraction = (boundary - first[1]) / (second[1] - first[1])
        return np.array((first[0] + fraction * (second[0] - first[0]), boundary))

    vertices = clip_edge(vertices, lambda point: point[0] >= 0, lambda first, second: intersection_x(first, second, 0))
    vertices = clip_edge(vertices, lambda point: point[0] <= width, lambda first, second: intersection_x(first, second, width))
    vertices = clip_edge(vertices, lambda point: point[1] >= 0, lambda first, second: intersection_y(first, second, 0))
    vertices = clip_edge(vertices, lambda point: point[1] <= height, lambda first, second: intersection_y(first, second, height))
    return vertices.astype(np.float32)


def select_visible_contours(
    contours: list[np.ndarray],
    image_shape: tuple[int, int],
    min_visible_area_fraction: float = MIN_VISIBLE_AREA_FRACTION,
) -> tuple[list[np.ndarray], list[float]]:
    """Keep contours whose in-image polygon area satisfies the configured threshold."""
    kept = []
    visible_fractions = []
    for contour in contours:
        full_area = polygon_area(contour)
        clipped_area = polygon_area(clip_polygon_to_image(contour, image_shape))
        fraction = 0.0 if full_area == 0 else clipped_area / full_area
        visible_fractions.append(fraction)
        if fraction >= min_visible_area_fraction:
            kept.append(contour)
    return kept, visible_fractions


class HAICUBubbleCPNDataset(Dataset):
    """One independent camera image per sample with overlap-aware CPN targets."""

    def __init__(
        self,
        data_root: str | Path,
        case_ids: Iterable[int],
        *,
        samples: int = 64,
        order: int = 6,
        max_bg_dist: float = 0.8,
        min_fg_dist: float = 0.85,
        min_visible_area_fraction: float = MIN_VISIBLE_AREA_FRACTION,
    ):
        self.data_root = Path(data_root)
        self.case_ids = list(case_ids)
        self.min_visible_area_fraction = min_visible_area_fraction
        if not 0 < self.min_visible_area_fraction <= 1:
            raise ValueError("min_visible_area_fraction must be in (0, 1]")
        self.records = [
            record
            for case_id in self.case_ids
            for record in list_case_records(self.data_root, case_id)
        ]
        self.case_indices: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.case_indices[record["case_id"]].append(index)

        self.target_generator = cd.data.CPNTargetGenerator(
            samples=samples,
            order=order,
            max_bg_dist=max_bg_dist,
            min_fg_dist=min_fg_dist,
        )
        self._cached_case_id: int | None = None
        self._annotations_by_image: dict[int, list[dict]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_case_annotations(self, case_id: int) -> None:
        if self._cached_case_id == case_id:
            return
        coco = load_case_coco(self.data_root, case_id)
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
        self._cached_case_id = case_id
        self._annotations_by_image = annotations_by_image

    def raw_sample(self, index: int) -> tuple[dict, np.ndarray, list[np.ndarray]]:
        """Return record, raw grayscale image, and COCO contours for inspection."""
        record = self.records[index]
        self._load_case_annotations(record["case_id"])
        image = iio.imread(self.data_root / record["file_name"])
        if image.ndim != 2:
            raise ValueError(f"Expected one grayscale channel: {record['file_name']}")
        if image.shape != (record["height"], record["width"]):
            raise ValueError(
                f"Image size mismatch for {record['file_name']}: {image.shape} != "
                f"{(record['height'], record['width'])}"
            )
        contours = annotation_contours(self._annotations_by_image[record["image_id"]])
        return record, image, contours

    def __getitem__(self, index: int) -> OrderedDict:
        record, image, contours = self.raw_sample(index)
        contours, _ = select_visible_contours(
            contours,
            image.shape,
            min_visible_area_fraction=self.min_visible_area_fraction,
        )
        labels = cd.data.contours2labels(
            [contour.copy() for contour in contours], size=image.shape, clip=True
        ).astype(np.int32)
        cd.data.relabel_(labels)

        generator = self.target_generator
        generator.feed(labels=labels)
        image = cd.data.normalize_percentile(image, percentile=99.8).astype(np.float32) / 255.0
        image = image[..., None]

        return OrderedDict(
            inputs=image,
            labels=generator.reduced_labels,
            fourier=(generator.fourier.astype(np.float32),),
            locations=(generator.locations.astype(np.float32),),
            sampled_contours=(generator.sampled_contours.astype(np.float32),),
            sampling=(generator.sampling.astype(np.float32),),
            targets=labels,
        )


class CaseBatchSampler(Sampler[list[int]]):
    """Shuffle batches while keeping every batch within one case directory."""

    def __init__(self, dataset: HAICUBubbleCPNDataset, batch_size: int, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        case_ids = list(self.dataset.case_indices)
        rng.shuffle(case_ids)
        for case_id in case_ids:
            indices = np.asarray(self.dataset.case_indices[case_id])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.dataset.case_indices.values()
        )
