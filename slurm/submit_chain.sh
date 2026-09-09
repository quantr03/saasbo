#!/bin/bash
# Submit an array and N re-submissions of it chained with --dependency=afterany, so every run in
# the array is resumed until it reaches T (README, "On SLURM"). afterany rather than afterok on
# purpose: a task the wall clock killed exits non-zero, and continuing it is what the next
# submission is for; a run already at T exits 0 in seconds.
#
#   bash slurm/submit_chain.sh <file.sbatch> [N_RESUBMITS=3] [sbatch options...]
#   bash slurm/submit_chain.sh slurm/sagp_product-lengthscale.sbatch 3
#   bash slurm/submit_chain.sh slurm/sagp_refs.sbatch 1 --array=0-9
#
# N_RESUBMITS must be given when sbatch options follow it. Runs from any directory: it changes to
# the repo root, where the sbatch files' --output=logs/... resolves.
set -euo pipefail

script="${1:?usage: submit_chain.sh <file.sbatch> [N_RESUBMITS=3] [sbatch options...]}"
n="${2:-3}"
shift $(( $# >= 2 ? 2 : 1 ))
script="$(cd "$(dirname "$script")" && pwd)/$(basename "$script")"

cd "$(dirname "$0")/.."
mkdir -p logs runs

jobid=$(sbatch --parsable "$@" "$script"); jobid=${jobid%%;*}
echo "$jobid  $(basename "$script")"
for _ in $(seq "$n"); do
    jobid=$(sbatch --parsable "$@" --dependency=afterany:"$jobid" "$script"); jobid=${jobid%%;*}
    echo "$jobid  $(basename "$script")  afterany the previous"
done
