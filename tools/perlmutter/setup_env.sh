#!/bin/bash
# Sets up the SolvationNet conda environment on Perlmutter.
#
# Usage:
#   bash setup_env.sh [env_name]
#
# Default env name: solvationnet
#
# This loads NERSC's conda module and creates an env with PyTorch (CUDA),
# ASE, and the ML potentials (orb-models, mace-torch, fairchem-core).
# Run once; re-run to update.

set -euo pipefail

ENV_NAME="${1:-solvationnet}"

echo "=== Setting up SolvationNet environment: $ENV_NAME ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

module load conda 2>/dev/null || module load python 2>/dev/null

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
    conda activate "$ENV_NAME"
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.12 -y
    conda activate "$ENV_NAME"
fi

echo "Installing PyTorch with CUDA support..."
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing core dependencies..."
pip install --upgrade \
    "ase>=3.22" \
    "numpy>=1.20" \
    "matplotlib>=3.5"

echo "Installing ML potentials..."
pip install --upgrade \
    orb-models \
    fairchem-core \
    mace-torch

echo ""
echo "=== Done ==="
echo "Activate with:  conda activate $ENV_NAME"
echo ""
echo "Test it:"
echo "  python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "  python -c \"from orb_models.forcefield import pretrained; print('orb-models OK')\""
