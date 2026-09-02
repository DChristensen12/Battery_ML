#!/bin/bash
# Builds the conda env for UMA OMOL MD on Perlmutter. Run once.
#
#   bash setup_env.sh [env_name]     # defaults to solvationnet
#
# The UMA weights are gated, so afterwards you also need:
#   python -c "from huggingface_hub import login; login()"

set -euo pipefail

ENV="${1:-solvationnet}"

echo "=== UMA OMOL environment setup: $ENV ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

module load conda 2>/dev/null || module load python 2>/dev/null

if conda env list | grep -q "^${ENV} "; then
    echo "Environment '$ENV' exists — updating..."
    conda activate "$ENV"
else
    echo "Creating '$ENV'..."
    conda create -n "$ENV" python=3.12 -y
    conda activate "$ENV"
fi

echo ""
echo "Installing PyTorch (CUDA 12.1)..."
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing ASE + FAIRChem..."
pip install --upgrade \
    "ase>=3.22" \
    "numpy>=1.20" \
    fairchem-core

echo ""
echo "=== Done ==="
echo ""
echo "Activate:   conda activate $ENV"
echo ""
echo "IMPORTANT — UMA model weights are gated on HuggingFace."
echo "Run once:   python -c \"from huggingface_hub import login; login()\""
echo ""
echo "Test:"
echo "  python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "  python -c \"from fairchem.core import pretrained_mlip; print('fairchem OK')\""
