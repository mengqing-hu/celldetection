#!/bin/bash -l
#SBATCH --job-name=haicu-cpn-full
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home2/hu09/project/celldetection/demos/haicu/outputs/log/slurm-%x-%j.out
#SBATCH --error=/data/home2/hu09/project/celldetection/demos/haicu/outputs/log/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=/data/home2/hu09/project/celldetection
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}/demos/haicu/code"

export PYTHONUNBUFFERED=1
export HAICU_TRAIN_CASES=1,2,3,4,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22,23,24,25
export HAICU_VALIDATION_CASES=5,10,11
export HAICU_EPOCHS=100
export HAICU_SCHEDULER_STEP_SIZE=25
export HAICU_EVAL_EVERY=5
export HAICU_ORDER=8
export HAICU_SAMPLES=128
export HAICU_MIN_VISIBLE_AREA_FRACTION=0.50
export HAICU_UNCERTAINTY_NMS=true
export HAICU_UNCERTAINTY_FACTOR=7.0

nvidia-smi
python --version
python train_bubble_coco.py
