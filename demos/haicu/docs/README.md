# HAICU bubble CPN experiment

This directory contains the code and documentation for training a Contour
Proposal Network (CPN) on the HAICU bubble dataset. The raw images and COCO
annotations are **not copied into the repository**; they remain in the shared
dataset directory:

```text
/bigdata/haicu/starke88/data/from_hendrik_hessenkemper_fwdc/bubble_multicamera_for_paper
```

## Current experiment

- Each camera image is an independent single-image 2D instance-segmentation
  sample. No multi-view fusion is used.
- The current scope is `Train_1`–`Train_25`.
- Training cases are `Train_1`–`Train_25`, excluding `Train_5`, `Train_10`,
  and `Train_11`.
- Validation cases are `Train_5`, `Train_10`, and `Train_11`.
- `Train_26` and all cases after `Train_30` are currently ignored.
- Ground truth comes from `coco_annotation_seg.json` and uses the full amodal
  polygon annotations.
- Boundary-crossing polygons are clipped to the image boundary and retained
  only when their visible-area fraction is at least `0.75`.
- Original image orientation and full `1920×640` resolution are preserved.
  No rotation, transpose, crop, or tile operation is performed.

## Code

- `../code/bubble_coco_dataset.py`: lazy COCO loader and CPN-target creation.
- `../code/bubble_coco_check.py`: annotation checks and target overlays.
- `../code/train_bubble_coco.py`: primary Config-based training, validation,
  threshold search, metrics, checkpoints, and visualizations.
- `../code/train_bubble_coco_pilot.sbatch`: Slurm pilot job using `Train_1`
  for training, `Train_5` for validation, and one epoch.
- `../code/train_bubble_coco_full.sbatch`: full current-scope run using all
  configured training cases and validation cases `5,10,11`.

## Execution logic

`train_bubble_coco.py` is the primary executable. It reads the COCO
`coco_annotation_seg.json` files lazily through `HAICUBubbleCPNDataset`, creates
CPN targets from the filtered polygons, and keeps batches from one case
together with `CaseBatchSampler`. The script then:

1. builds the configured CPN model;
2. trains with Adam, AMP, and a StepLR scheduler;
3. computes validation loss and instance metrics after every epoch;
4. saves the checkpoint with the best validation mean F1;
5. evaluates the selected checkpoint at the configured score/NMS thresholds;
6. saves IoU metrics, checkpoints, loss figures, and selected visualizations.

The current configuration uses `CpnU22`, `order=6`, `samples=64`, three contour
refinement iterations, six refinement buckets, `batch_size=1`, and
`num_workers=8`. It preserves the full image resolution and enables mixed
precision on CUDA. The uncertainty head and uncertainty-aware NMS are enabled.

The current threshold lists contain one pair only:

```python
val_score_threshs = (0.9,)
val_nms_threshs = (0.5,)
```

Consequently, the current run does not perform a broad threshold sweep; it
evaluates the fixed pair `(score=0.9, NMS=0.5)`. A broader sweep can be enabled
by adding more values to these two lists, but it requires repeated validation
passes and substantially increases runtime.

## Run a pilot

From the code directory, submit the pilot to an A100 node:

```bash
cd /home/hu09/project/celldetection/demos/haicu/code
sbatch train_bubble_coco_pilot.sbatch
```

The pilot script activates the repository virtual environment and sets:

```text
HAICU_TRAIN_CASES=1
HAICU_VALIDATION_CASES=5
HAICU_EPOCHS=1
```

Monitor the job with:

```bash
squeue --me
squeue -u "$USER"
tail -f ../outputs/log/slurm-haicu-cpn-pilot-<jobid>.out
```

For a direct run with different cases or epochs, set the same environment
variables before executing `python train_bubble_coco.py`. The default script
configuration uses all training cases listed above, validation cases `5,10,11`,
and 20 epochs.

## Run the full current scope

The primary script already defaults to the complete current split, so the full
Slurm job only needs to set the epoch count:

```bash
cd /home/hu09/project/celldetection/demos/haicu/code
sbatch train_bubble_coco_full.sbatch
```

Do not set `HAICU_TRAIN_CASES=1` or `HAICU_VALIDATION_CASES=5` for this run;
those overrides are only used by the pilot job.

## Evaluation and outputs

The script evaluates every validation image numerically. It searches the
configured score/NMS threshold combinations, selects the best mean F1, and
saves metrics at IoU thresholds `0.5`–`0.9`. The uncertainty Spearman/AUROC,
risk-coverage, and uncertainty-error scatter outputs are intentionally not
generated.

Each run creates `../outputs/run_<timestamp>_<id>/` containing:

- `config.json`, `split.json`: configuration and case split;
- `best_model.pt`, `selected_model.pt`: checkpoints;
- `validation_summary.json`: final metrics and selected thresholds;
- `figures/`: loss curves, CPN target examples, and IoU/count plots;
- `validation_predictions/`: predictions for 10 deterministic random
  validation images, with ground-truth contours, predicted contours, score
  (`S`), and mean box uncertainty (`U`);
- `uncertainty/`: full-image uncertainty overlays for the same 10 images,
  including every predicted bubble's box and four edge uncertainties;
- `uncertainty/top_instances/`: the ten highest-uncertainty individual
  instances for focused inspection.

Output filenames preserve the source image path. For example,
`Train_5_Cam1__0.png` corresponds to `Train_5/Cam1/_0.png`.

The numerical evaluation still processes all validation images; only the
saved visualizations are limited to 10 images to reduce runtime and disk use.

