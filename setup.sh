#!/bin/bash
# ── setup.sh ───────────────────────────────────────────────────
# Run this ONCE on the cluster to set up the environment.
# Usage:  bash setup.sh
# ───────────────────────────────────────────────────────────────

set -e

echo "Setting up ssn-speech environment..."

# install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# create venv and install dependencies
echo "Creating virtual environment..."
uv venv --python 3.10

echo "Installing PyTorch with CUDA 12.1..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing other dependencies..."
uv pip install numpy matplotlib scikit-learn tqdm

# create directories
mkdir -p logs results figures

echo ""
echo "Setup complete. To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run experiments:"
echo "  sbatch submit_seeds.sh"
