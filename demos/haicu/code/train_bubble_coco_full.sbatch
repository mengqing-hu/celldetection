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
export HAICU_EPOCHS=20

nvidia-smi
python --version
python train_bubble_coco.py
