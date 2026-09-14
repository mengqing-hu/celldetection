#!/bin/bash -l
#SBATCH --job-name=haicu-cpn-uncertainty-eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home2/hu09/project/celldetection/demos/haicu/outputs/log/slurm-%x-%j.out
#SBATCH --error=/data/home2/hu09/project/celldetection/demos/haicu/outputs/log/slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=/data/home2/hu09/project/celldetection
source "${PROJECT_DIR}/.venv/bin/activate"
cd "${PROJECT_DIR}/demos/haicu/code"

export PYTHONUNBUFFERED=1

# Usage: sbatch evaluate_uncertainty.sbatch <run_dir> [extra evaluate_uncertainty.py args...]
# Example:
#   sbatch evaluate_uncertainty.sbatch ../outputs/run_20260909_193730_5c1fa59f
RUN_DIR=${1:?"pass the run directory, e.g. ../outputs/run_20260909_193730_5c1fa59f"}
shift || true

nvidia-smi
python --version
python evaluate_uncertainty.py "${RUN_DIR}" --score-thresh 0.1 "$@"
