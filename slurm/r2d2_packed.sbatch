#!/bin/bash
#SBATCH --job-name=r2d2-packed
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --gpus=1
# One task: the K runs are background processes of this batch script, not srun steps, so they all
# share the task's one GPU and its cores (one core each; torch and XLA are pinned to one thread).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Memory: 8 x 12.3 GB measured MaxRSS = 98 GB, requested 160G (four complete pilot runs at t = 199
# peaked at 11.9-12.3 GB each, and the K runs start together, so they peak together); each GPU's
# share of an H200 node's 1.98 TB is about 250 GB. On the GPU a run holds about 4.8 GB, so
# 8 x 5 GB = 40 GB of the H200's 141 GB. 16 cores per GPU is the node's 128 cores over its 8 GPUs.
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# K BO runs on one GPU at the same time, under NVIDIA MPS (Quan, 2026-10-01): replaces the
# one-run-per-GPU cell arrays (slurm/r2d2_<cell>.sbatch) for the rest of the R2-D2 study. What a run
# computes is unchanged; only how many runs share a GPU.
#
# Measured (job 20586178, H200, additive/amplitude_r2d2 at T = 28): one run saturates its core and keeps the GPU busy 42 % of the time (fit 23 s of a 168 s iteration; the rest is SciPy's L-BFGS-B acquisition on the CPU).
# Per run, K = 4 with MPS 1.30x slower at 3.05x the throughput; K = 8 with MPS 1.69x slower at 4.67x the throughput, GPU 93 % busy; K = 4 without MPS 2.4x slower (fits 9x slower), 1.6x the throughput.
# Why MPS: packing overlaps one run's CPU-bound acquisition with another's GPU fit. Without MPS the
# GPU time-slices between the K processes' contexts, so concurrent fits take turns instead of
# sharing the card; MPS runs the K processes' kernels in one context, side by side. K = 8 is the
# choice. The daemon is started and stopped per task, with pipe and log directories of its own.
#
# WALL TIMES: a packed run's wall times (fit_wall_s and the other timings it records) are about
# 1.7x an unpacked run's and depend on what else shares its GPU. Never compare them with unpacked
# runs' wall times; a cost table over both needs a runs-per-GPU column.
#
# Input: WORK (required), a work list file with one run per line, "<cell> <family> <seed>", e.g.
#   additive/lengthscale_r2d2 aligned10 3
# Blank lines and # comments are skipped; a relative WORK is read from REPO. Array task i runs the
# list's runs K*i .. K*i+K-1 (K, default 8); a task past the end exits 2. So there is no default
# --array: the caller passes --array=0-<n> with n = ceil(runs / K) - 1. OUT (default runs_r2d2/),
# T (200), NUTS (512,256,16) and REPO ($WRKDIR/saasbo) as in the cell launchers.
# Each run logs to logs/r2d2-packed_<array job>_<task>.<cell with / as ->.<family>.seed<NN>.log;
# this file (%x_%A_%a.out) gets one start and one end line per run. The task exits 1 if any run
# failed, and 0 when all K reached T.
#
# Re-submitting the same array resumes every run in it (each from its own checkpoint; a run
# already at T exits in seconds), so --time is a checkpoint interval, not a deadline. Chain it:
#   bash slurm/submit_chain.sh slurm/r2d2_packed.sbatch 3 --export=ALL,WORK=<file> --array=0-<n>
#   bash slurm/submit_chain.sh slurm/r2d2_packed.sbatch 3 \
#       --export=ALL,WORK=slurm/work/pilot-remaining-2026-10-01.txt --array=0-2
# A run must appear once in all the work lists live at a time: two processes on one run directory
# corrupt it. Submit from the repo root so logs/ resolves, or use slurm/submit_chain.sh.
set -eo pipefail

REPO="${REPO:-$WRKDIR/saasbo}"
OUT="${OUT:-runs_r2d2/}"
T="${T:-200}"
NUTS="${NUTS:-512,256,16}"    # one NUTSConfig for all eight cells; a change is preregistered
K="${K:-8}"
WORK="${WORK:?set WORK to a work list file, one '<cell> <family> <seed>' per line}"

# One core per run: torch reads OMP_NUM_THREADS at start-up (sagp/__init__.py copies it into
# torch.set_num_threads) and XLA reads XLA_FLAGS; both must be set before Python starts.
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# torch 2.14 sends a few CUDA ops (a batched outer product in the acquisition's backward pass) to
# Triton kernels that compile a C launcher on first use, and the GPU nodes have no C compiler; this
# keeps torch on its stock CUDA kernels. torch reads it once, so it too is set before Python starts.
export TORCH_DISABLE_NATIVE_JIT=1
# K JAX processes share the card: without this each would try to preallocate most of its memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
module load mamba
source activate saasbo        # conda's activate trips over set -u, hence set -u only afterwards
set -u

i=${SLURM_ARRAY_TASK_ID:?submit with --array=0-<n>; this launcher has no default array}
cd "$REPO"
mkdir -p logs

runs=()
n=0
while IFS= read -r line || [ -n "$line" ]; do
    n=$(( n + 1 ))
    line=${line%%#*}
    cell='' family='' seed='' extra=''
    read -r cell family seed extra <<< "$line" || true
    [ -z "$cell" ] && continue
    if [ -z "$seed" ] || [ -n "$extra" ] || ! [[ $seed =~ ^[0-9]+$ ]]; then
        echo "$WORK:$n: expected '<cell> <family> <seed>', got '$line'" >&2
        exit 1
    fi
    runs+=("$cell $family $(( 10#$seed ))")
done < "$WORK"
dups=$(printf '%s\n' "${runs[@]}" | sort | uniq -d)
if [ -n "$dups" ]; then
    echo "$WORK lists a run twice (two processes would write one run directory): $dups" >&2
    exit 1
fi

first=$(( K * i ))
if (( first >= ${#runs[@]} )); then
    echo "task $i has no work: $WORK lists ${#runs[@]} runs at K = $K per task, so the array is 0-$(( (${#runs[@]} + K - 1) / K - 1 ))" >&2
    exit 2
fi
mine=("${runs[@]:first:K}")
echo "task $i on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-unset}: runs $first-$(( first + ${#mine[@]} - 1 )) of $WORK (${#runs[@]} runs, K = $K), $(date '+%F %T')"

if ! command -v nvidia-cuda-mps-control > /dev/null; then
    echo "no nvidia-cuda-mps-control on $(hostname): without MPS a packed run is 2.4x slower; not starting" >&2
    exit 1
fi
MPS_ROOT="${TMPDIR:-$REPO/logs}/mps_${SLURM_JOB_ID}"
export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# The daemon is stopped on every way out: normal end, a failed run, and the SIGTERM Slurm sends at
# the time limit or on scancel (the runs are sent SIGTERM too; each resumes from its checkpoint).
mps_up=0
pids=()
stop_mps() {
    if [ "$mps_up" = 1 ]; then
        echo quit | nvidia-cuda-mps-control || echo "MPS daemon did not acknowledge quit" >&2
        mps_up=0
    fi
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
}
on_signal() {   # stop the runs (each python is a child of its run_one subshell), then exit
    echo "signal received: stopping the ${#pids[@]} runs, $(date '+%F %T')" >&2
    if (( ${#pids[@]} )); then
        pkill -TERM -P "$(IFS=,; echo "${pids[*]}")" 2> /dev/null || true
        kill "${pids[@]}" 2> /dev/null || true
    fi
    exit "$1"
}
trap stop_mps EXIT
trap 'on_signal 143' TERM
trap 'on_signal 130' INT
nvidia-cuda-mps-control -d
mps_up=1
sleep 2
echo "MPS daemon started, pipe $CUDA_MPS_PIPE_DIRECTORY"

run_one() {   # cell family seed log
    local cell=$1 family=$2 seed=$3 log=$4 t0 rc=0
    t0=$(date +%s)
    python -m experiments.run_bo --family "$family" --seed "$seed" --cell "$cell" \
        --T "$T" --out "$OUT" --objective-dir data/objectives --nuts "$NUTS" > "$log" 2>&1 || rc=$?
    echo "end   family=$family seed=$seed cell=$cell exit=$rc wall=$(( $(date +%s) - t0 ))s $(date '+%F %T')"
    return $rc
}

for run in "${mine[@]}"; do
    read -r cell family seed <<< "$run"
    log="logs/r2d2-packed_${SLURM_ARRAY_JOB_ID}_${i}.${cell//\//-}.${family}.seed$(printf %02d "$seed").log"
    echo "start family=$family seed=$seed cell=$cell log=$log $(date '+%F %T')"
    run_one "$cell" "$family" "$seed" "$log" &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
pids=()
echo "task $i done: ${#mine[@]} runs, exit $status, $(date '+%F %T')"
exit $status
