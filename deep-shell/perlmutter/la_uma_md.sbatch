#!/bin/bash
#SBATCH --job-name=la_uma_md
#SBATCH --account=M4292
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=la_uma_md_%j.out
#SBATCH --error=la_uma_md_%j.err

# All five anion systems at once on four A100s, round-robin, so PO4 ends up
# sharing GPU 0 with F. A 390-atom molecule barely dents an 80 GB card and
# sharing one costs far less than leaving a system to never start at all.
#
# 10 ns at 1 fs is 10M UMA calls per system. That will not finish in a single
# 24 h allocation, so resubmit this until all five say DONE. Everything
# checkpoints and resumes. Running them concurrently is what keeps them
# advancing together across resubmissions.
#
# Submit from $SCRATCH/solvationnet/deep-shell/:
#   sbatch perlmutter/la_uma_md.sbatch

set -uo pipefail

module load conda 2>/dev/null || true
conda activate solvationnet || { echo "ERROR: conda activate solvationnet failed"; exit 1; }
python -c "import fairchem, torch; assert torch.cuda.is_available()" \
    || { echo "ERROR: fairchem/CUDA unavailable in this environment"; exit 1; }

WORK="${SLURM_SUBMIT_DIR:-$SCRATCH/solvationnet/deep-shell}"
cd "$WORK" || exit 1

echo "=== La³⁺ solvation MD ==="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Dir:  $(pwd)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

SYSTEMS=(F OH NO3 CO3 PO4)
NGPU=4

mkdir -p results
for sys in "${SYSTEMS[@]}"; do
    mkdir -p "results/La3+_${sys}"
done

echo "Launching ${#SYSTEMS[@]} systems across $NGPU GPUs"
PIDS=()
NAMES=()
for i in "${!SYSTEMS[@]}"; do
    sys="${SYSTEMS[$i]}"
    gpu=$(( i % NGPU ))
    echo "  GPU $gpu → La3+_${sys}"
    CUDA_VISIBLE_DEVICES=$gpu python run_uma_md.py \
        "droplets/La3+_${sys}_droplet.xyz" \
        -o "results/La3+_${sys}/" \
        --resume \
        > "results/La3+_${sys}/slurm.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$sys")
done
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  La3+_${NAMES[$i]}: exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done
echo "All runs returned ($(date)), failures: $FAIL"

echo ""
echo "=== Summary ==="
INCOMPLETE=0
for sys in "${SYSTEMS[@]}"; do
    dir="results/La3+_${sys}"
    if [ -f "$dir/final.xyz" ]; then
        echo "  La3+_${sys}:  DONE (final.xyz exists)"
    elif [ -f "$dir/checkpoint.pkl" ]; then
        step=$(tail -n 1 "$dir/md.log" 2>/dev/null | awk '{print $1}')
        echo "  La3+_${sys}:  PARTIAL (last logged step: ${step:-unknown}) — resubmit to continue"
        INCOMPLETE=$((INCOMPLETE + 1))
    else
        echo "  La3+_${sys}:  FAILED (no output; see $dir/slurm.log)"
        INCOMPLETE=$((INCOMPLETE + 1))
    fi
done

if [ "$INCOMPLETE" -gt 0 ]; then
    echo ""
    echo "$INCOMPLETE system(s) incomplete — resubmit: sbatch perlmutter/la_uma_md.sbatch"
fi

echo ""
echo "Finished: $(date)"
# clamped, since FAIL + a child status can wrap past 255 and read as success
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
