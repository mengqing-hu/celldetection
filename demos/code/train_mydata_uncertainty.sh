#!/bin/bash
#SBATCH --job-name=cpn-uncertainty
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load python/3.12.4 cuda/12.8

PROJECT=/data/home2/hu09/project/celldetection
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MLFLOW_TRACKING_URI="sqlite:///$PROJECT/mlruns/mlflow.db"
export MLFLOW_EXPERIMENT_NAME="celldetection-mydata-uncertainty"

cd "$PROJECT/demos/code"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
START=$(date +%s)
echo "Start time: $(date)"

python - <<'PY'
import torch
import celldetection as cd
import mlflow

print("torch:", torch.__version__)
print("celldetection:", cd.__version__)
print("mlflow:", mlflow.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo "MLflow experiment: $MLFLOW_EXPERIMENT_NAME"

python mydata_uncertainty.py

END=$(date +%s)
echo "End time: $(date)"
echo "Duration: $(( (END - START) / 60 )) minutes"
