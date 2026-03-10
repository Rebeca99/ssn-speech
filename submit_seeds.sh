#!/bin/bash
#SBATCH --job-name=ssn_seeds
#SBATCH --array=0-4
#SBATCH --partition=gpu          # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seed_%a.out
#SBATCH --error=logs/seed_%a.err

# ── setup ──────────────────────────────────────────────────────
# Adjust PROJECT_DIR for your cluster.

PROJECT_DIR="$HOME/ssn-speech"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p results

# ── run ────────────────────────────────────────────────────────
SEED=$SLURM_ARRAY_TASK_ID

echo "Running seed $SEED on $(hostname) with GPU $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python run_seed.py \
    --seed $SEED \
    --outdir results/seed_${SEED} \
    --epochs_baseline 12 \
    --epochs_ssn 15 \
    --n_train 60000 \
    --n_val 8000 \
    --n_eval 3000 \
    --n_trace 2000

echo "End time: $(date)"
echo "Done with seed $SEED"
