#!/bin/bash
#SBATCH --job-name=sagp2-product-amplitude
#SBATCH --array=0-39
# One task, stated: a node that hands out CPUs in hyperthread pairs (the GPU nodes) gives a task 2,
# and srun would otherwise start one copy of the run per CPU, both writing the same run directory.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 32G, not the first study's 8G: host memory grows 70-100 MB per BO iteration (about 20 GB by
# t=200), and the 8G tasks of the first study were OOM-killed around t=88.
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gpus=1
# One H200 per task, on the ELLIS partition and account: sagp runs the chain and the acquisition
# on it (sagp/__init__.py). A GPU run is not bit-identical to itself run again, even on the same
# device (README), so unlike a CPU run there is no hardware to pin it to.
#
# The second family set of the product/amplitude cell: one (family, seed) per array task,
# 4 families x 10 seeds = 40 tasks; slurm/sagp_product-amplitude.sbatch is the first set.
# Re-submitting the same array resumes every run in it: --time is a checkpoint interval, not a
# deadline (README, "On SLURM"), sized to this cell's cost. Submit from the repo root so logs/
# resolves, or use slurm/submit_chain.sh.
# Smoke run: sbatch --export=ALL,OUT=runs_smoke/,T=22 --array=0 slurm/sagp2_product-amplitude.sbatch
set -eo pipefail

CELL=product/amplitude
REPO="${REPO:-$WRKDIR/saasbo}"
OUT="${OUT:-runs/}"
T="${T:-200}"
NUTS="${NUTS:-512,256,16}"    # the preregistered fallback is 128,128,8, identical for every cell
FAMILIES=(interaction_g0.00 interaction_g0.50 interaction_g0.75 anti_aligned)

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

i=$SLURM_ARRAY_TASK_ID; seed=$((i % 10)); f=$(( i / 10 ))
cd "$REPO"
srun python -m experiments.run_bo --family "${FAMILIES[$f]}" --seed "$seed" --cell "$CELL" \
    --T "$T" --out "$OUT" --objective-dir data/objectives --nuts "$NUTS"
