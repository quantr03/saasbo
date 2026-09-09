#!/bin/bash
#SBATCH --job-name=sagp-refs
#SBATCH --array=0-119
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Pin one CPU family so a resumed run stays bit-identical to its earlier iterations (README:
# a resume on a different CPU differs at the ulp level); off by default so any node will do.
##SBATCH --constraint=csl
#
# One (family, seed, reference) per array task, 4 families x 10 seeds x 3 references = 120 tasks.
# Re-submitting the same array resumes every run in it: --time is a checkpoint interval, not a
# deadline (README, "On SLURM"). Submit from the repo root so logs/ resolves, or use
# slurm/submit_chain.sh. Smoke run: sbatch --export=ALL,OUT=runs_smoke/,T=22 --array=0 slurm/sagp_refs.sbatch
set -eo pipefail

REPO="${REPO:-$WRKDIR/saasbo}"
OUT="${OUT:-runs/}"
T="${T:-200}"
FAMILIES=(aligned3 aligned10 decoupled interaction_g0.25)
REFS=(sobol dsp_map oracle_S)

# An array task gets one core: torch reads OMP_NUM_THREADS at start-up (sagp/__init__.py copies
# it into torch.set_num_threads) and XLA reads XLA_FLAGS; both must be set before Python starts.
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
module load mamba
source activate saasbo        # conda's activate trips over set -u, hence set -u only afterwards
set -u

i=$SLURM_ARRAY_TASK_ID; seed=$((i % 10)); r=$(( (i / 10) % 3 )); f=$(( i / 30 ))
cd "$REPO"
srun python -m experiments.run_bo --family "${FAMILIES[$f]}" --seed "$seed" --cell "${REFS[$r]}" \
    --T "$T" --out "$OUT" --objective-dir data/objectives
