#!/bin/bash
#SBATCH --job-name=r2d2-product-lengthscale
#SBATCH --array=0-79
# One task, stated: a node that hands out CPUs in hyperthread pairs (the GPU nodes) gives a task 2,
# and srun would otherwise start one copy of the run per CPU, both writing the same run directory.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 32G, as the half-Cauchy cells: host memory grows 70-100 MB per BO iteration (about 20 GB by
# t=200).
#SBATCH --mem=32G
# Two days for every R2-D2 cell (plan D10): the 24 h limit was exceeded at 26.6 h and 25.7 h, and
# the R2-D2 cost is not yet measured on GPU.
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --gpus=1
# One H200 per task, on gpu-h200-141g-short with no account: Quan's instruction of 2026-09-28 not
# to use ELLIS for now (ruling R37). Its 2-day limit covers --time; every R2-D2 run then runs on
# the same GPU type as the replay, so wall times are comparable. ELLIS jobs have priority on these
# nodes, so tasks may queue.
#
# The product/lengthscale_r2d2 cell (plan D10): one (family, seed) per array task; task i runs seed i % NSEEDS of
# family FAMILIES[i / NSEEDS]. FAMILIES (space-separated, default all eight) and NSEEDS (default
# 10) are read from the environment:
#   pilot:     sbatch --export=ALL,FAMILIES="aligned10 decoupled",NSEEDS=5 --array=0-9 slurm/r2d2_product-lengthscale.sbatch
#   replicate: sbatch --array=0-79 slurm/r2d2_product-lengthscale.sbatch
# Re-submitting the same array resumes every run in it: --time is a checkpoint interval, not a
# deadline (README, "On SLURM"); chain the resumes with slurm/submit_chain.sh <this file> 3. Submit
# from the repo root so logs/ resolves, or use slurm/submit_chain.sh.
# Smoke run: sbatch --export=ALL,OUT=runs_smoke/,T=22 --array=0 slurm/r2d2_product-lengthscale.sbatch
set -eo pipefail

CELL=product/lengthscale_r2d2
REPO="${REPO:-$WRKDIR/saasbo}"
OUT="${OUT:-runs_r2d2/}"
T="${T:-200}"
NUTS="${NUTS:-512,256,16}"    # one NUTSConfig for all eight cells; a change is preregistered
read -ra FAMS <<< "${FAMILIES:-aligned10 decoupled aligned3 anti_aligned interaction_g0.00 interaction_g0.25 interaction_g0.50 interaction_g0.75}"
NSEEDS="${NSEEDS:-10}"

# An array task gets one core: torch reads OMP_NUM_THREADS at start-up (sagp/__init__.py copies
# it into torch.set_num_threads) and XLA reads XLA_FLAGS; both must be set before Python starts.
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# torch 2.14 sends a few CUDA ops (a batched outer product in the acquisition's backward pass) to
# Triton kernels that compile a C launcher on first use, and the GPU nodes have no C compiler; this
# keeps torch on its stock CUDA kernels. torch reads it once, so it too is set before Python starts.
export TORCH_DISABLE_NATIVE_JIT=1
module load mamba
source activate saasbo        # conda's activate trips over set -u, hence set -u only afterwards
set -u

i=$SLURM_ARRAY_TASK_ID; seed=$(( i % NSEEDS )); f=$(( i / NSEEDS ))
if (( f >= ${#FAMS[@]} )); then
    echo "task $i has no work: ${#FAMS[@]} families x $NSEEDS seeds is 0-$(( ${#FAMS[@]} * NSEEDS - 1 ))" >&2
    exit 2
fi
cd "$REPO"
srun python -m experiments.run_bo --family "${FAMS[$f]}" --seed "$seed" --cell "$CELL" \
    --T "$T" --out "$OUT" --objective-dir data/objectives --nuts "$NUTS"
