#!/bin/bash
# ── run_all_sequential.sh ──────────────────────────────────────
# Run all 5 seeds one after another on a single GPU.
# Use this if you don't have SLURM (e.g., single A100 machine).
#
# Usage:  bash run_all_sequential.sh
# ───────────────────────────────────────────────────────────────

set -e

source .venv/bin/activate
mkdir -p results logs

for SEED in 0 1 2 3 4; do
    echo "════════════════════════════════════════════════════"
    echo "  SEED $SEED — $(date)"
    echo "════════════════════════════════════════════════════"

    python run_seed.py \
        --seed $SEED \
        --outdir results/seed_${SEED} \
        --epochs_baseline 12 \
        --epochs_ssn 15 \
        --n_train 60000 \
        --n_val 8000 \
        --n_eval 3000 \
        --n_trace 2000 \
        2>&1 | tee logs/seed_${SEED}.log

    echo "  Seed $SEED done at $(date)"
    echo ""
done

echo "════════════════════════════════════════════════════"
echo "  All seeds complete. Aggregating..."
echo "════════════════════════════════════════════════════"

python aggregate_seeds.py --results_dir results --n_seeds 5 --outdir figures

echo "Done. Figures in ./figures/"
