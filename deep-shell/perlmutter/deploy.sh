#!/bin/bash
# Push the scripts and droplets up to Perlmutter. Run it from the repo root:
#   bash deep-shell/perlmutter/deploy.sh <nersc_username>
#
# Then ssh over and:
#   cd $SCRATCH/solvationnet/deep-shell
#   bash perlmutter/setup_env.sh          # once
#   sbatch perlmutter/la_uma_md.sbatch

set -euo pipefail

USER="${1:?Usage: $0 <nersc_username>}"
HOST="perlmutter.nersc.gov"
REMOTE="$USER@$HOST"

SCRATCH_DIR="\$SCRATCH/solvationnet/deep-shell"
echo "Deploying to $REMOTE:$SCRATCH_DIR"

echo ""
echo "Creating remote directories..."
ssh "$REMOTE" "mkdir -p $SCRATCH_DIR/droplets $SCRATCH_DIR/perlmutter $SCRATCH_DIR/results"

echo ""
echo "Uploading scripts..."
scp deep-shell/run_uma_md.py "$REMOTE:$SCRATCH_DIR/"
scp deep-shell/build_droplet.py "$REMOTE:$SCRATCH_DIR/"
scp deep-shell/perlmutter/la_uma_md.sbatch "$REMOTE:$SCRATCH_DIR/perlmutter/"
scp deep-shell/perlmutter/setup_env.sh "$REMOTE:$SCRATCH_DIR/perlmutter/"

echo ""
echo "Uploading droplets..."
DROPLET_DIR=""
if [ -d "droplets" ]; then
    DROPLET_DIR="droplets"
elif [ -d "data/La3+_Droplets" ]; then
    DROPLET_DIR="data/La3+_Droplets"
fi

if [ -n "$DROPLET_DIR" ]; then
    scp "$DROPLET_DIR"/La3+_*_droplet.xyz "$REMOTE:$SCRATCH_DIR/droplets/"
    echo "  Uploaded from $DROPLET_DIR/"
else
    echo "  WARNING: No droplet directory found."
    echo "  Upload manually: scp La3+_*_droplet.xyz $REMOTE:$SCRATCH_DIR/droplets/"
fi

echo ""
echo "Done. Next steps on Perlmutter:"
echo ""
echo "  ssh $REMOTE"
echo "  cd $SCRATCH_DIR"
echo "  bash perlmutter/setup_env.sh          # one-time env setup"
echo "  sbatch perlmutter/la_uma_md.sbatch    # submit the job"
echo "  squeue -u \$USER                       # check status"
