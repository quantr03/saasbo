#!/bin/bash
#SBATCH --job-name=r2d2-replay
#SBATCH --array=0-39
# One task, stated: a node that hands out CPUs in hyperthread pairs (the GPU nodes) gives a task 2,
# and srun would otherwise start one copy of each replay per CPU, both writing the same directory.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 32G as the cell arrays: a replay process fits three t of one seed and exits, so it never grows
# the way a 180-iteration run does, and this is headroom rather than a measured need.
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gpus=1
# One H200 per task, on the ELLIS partition and account, as every R2-D2 array (plan D10): the
# replay's wall times are compared with its own control refits, so all of them run on one GPU type.
#
# Offline replay (plan D9, Task 11): refit a cell on a stored run's data at t = 50, 100 and 199
# (python -m experiments.replay fit). Forty tasks, each looping the ten seeds:
#   tasks 0-31   the four R2-D2 cells x the eight families, each on its half-Cauchy twin's runs:
#                cell R2D2_CELLS[i % 4], family FAMILIES[i / 4];
#   tasks 32-39  the four half-Cauchy cells x {aligned10, decoupled}, each on its own runs (the
#                control): cell HC_CELLS[j % 4], family CONTROL_FAMILIES[j / 4], j = i - 32.
# FAMILIES lists the pilot's two families first, so tasks 0-7 and 32-39 are the R2-D2 fits and
# their controls on the same datasets. Re-submitting the array resumes it: every t with a complete
# row in replay.csv is skipped, so --time is a checkpoint interval, not a deadline. A seed whose
# replay fails does not stop the task's other seeds; the task then exits 1, and a re-submission
# retries only what is missing.
#
# GPU cost probe (Task 11, before the bulk array), t = 199 of aligned10 seed 0 for all eight cells
# into runs_smoke/replay_probe/:
#   sbatch --export=ALL,PROBE=1 --array=0 slurm/r2d2_replay.sbatch
# Submit from the repo root so logs/ resolves. RUNS and OUT override the roots read and written.
set -eo pipefail

REPO="${REPO:-$WRKDIR/saasbo}"
RUNS="${RUNS:-runs/}"
OUT="${OUT:-runs_replay/}"
PROBE="${PROBE:-0}"
R2D2_CELLS=(additive/amplitude_r2d2 product/amplitude_r2d2 additive/lengthscale_r2d2 product/lengthscale_r2d2)
HC_CELLS=(additive/amplitude product/amplitude additive/lengthscale product/lengthscale)
FAMILIES=(aligned10 decoupled aligned3 anti_aligned interaction_g0.00 interaction_g0.25 interaction_g0.50 interaction_g0.75)
CONTROL_FAMILIES=(aligned10 decoupled)

# An array task gets one core: torch reads OMP_NUM_THREADS at start-up (sagp/__init__.py copies
# it into torch.set_num_threads) and XLA reads XLA_FLAGS; both must be set before Python starts.
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# torch 2.14 sends a few CUDA ops to Triton kernels that compile a C launcher on first use, and the
# GPU nodes have no C compiler; this keeps torch on its stock CUDA kernels. torch reads it once, so
# it too is set before Python starts.
export TORCH_DISABLE_NATIVE_JIT=1
module load mamba
source activate saasbo        # conda's activate trips over set -u, hence set -u only afterwards
set -u

i=$SLURM_ARRAY_TASK_ID
cd "$REPO"

if [ "$PROBE" = 1 ]; then
    # One probe, not one per task: forty copies would write the same eight directories at once.
    if [ "$i" != 0 ]; then
        echo "PROBE=1 runs as task 0 only (submit with --array=0); task $i does nothing" >&2
        exit 0
    fi
    status=0
    for cell in "${R2D2_CELLS[@]}" "${HC_CELLS[@]}"; do
        srun python -m experiments.replay fit --runs "$RUNS" --out runs_smoke/replay_probe/ \
            --family aligned10 --seed 0 --cell "$cell" --t 199 || status=1
    done
    exit $status
fi

if (( i < 32 )); then
    cell=${R2D2_CELLS[i % 4]}; family=${FAMILIES[i / 4]}
elif (( i < 40 )); then
    j=$(( i - 32 )); cell=${HC_CELLS[j % 4]}; family=${CONTROL_FAMILIES[j / 4]}
else
    echo "task $i has no work: the array is 0-39" >&2
    exit 2
fi

status=0
for seed in 0 1 2 3 4 5 6 7 8 9; do
    srun python -m experiments.replay fit --runs "$RUNS" --out "$OUT" --family "$family" \
        --seed "$seed" --cell "$cell" --t 50,100,199 || status=1
done
exit $status
