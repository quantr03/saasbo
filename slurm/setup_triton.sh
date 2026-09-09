#!/bin/bash
# One-time setup on an Aalto Triton login node: the checkout under $WRKDIR, the conda environment,
# the objective grid the array tasks share, and a dry run. Idempotent: re-run it to pull the
# branch or to repair the environment.
#
#   REPO_URL=git@github.com:<you>/saasbo.git bash slurm/setup_triton.sh
#
# Optional overrides: BRANCH (feat/botorch-saasbo), REPO_DIR ($WRKDIR/saasbo), ENV_NAME (saasbo).
set -eo pipefail

REPO_URL="${REPO_URL:?set REPO_URL to the git remote that carries the branch}"
BRANCH="${BRANCH:-feat/botorch-saasbo}"
REPO_DIR="${REPO_DIR:-$WRKDIR/saasbo}"
ENV_NAME="${ENV_NAME:-saasbo}"

# Triton's conda recipe: packages and environments live in the work directory (200 GB), not in the
# 10 GB home; pip's download cache goes there too, since torch alone is gigabytes.
module load mamba
mkdir -p "$WRKDIR/.conda_pkgs" "$WRKDIR/.conda_envs" "$WRKDIR/.pip_cache"
conda config --prepend pkgs_dirs "$WRKDIR/.conda_pkgs"
conda config --prepend envs_dirs "$WRKDIR/.conda_envs"
export PIP_CACHE_DIR="$WRKDIR/.pip_cache"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# Python 3.11 plus the pins in requirements-sagp.txt (torch from PyPI, so the pin test holds).
if [ ! -d "$WRKDIR/.conda_envs/$ENV_NAME" ]; then
    mamba create -y -n "$ENV_NAME" python=3.11 pip
fi
source activate "$ENV_NAME"   # conda's activate trips over set -u, hence set -u only afterwards
set -u
pip install -r requirements-sagp.txt pytest

# data/ is git-ignored, so the grid every array task loads through --objective-dir is built here
# (minutes; existing stems are skipped).
if [ ! -f data/objectives/manifest.json ]; then
    python -m synthobj.generate --out data/objectives
fi
mkdir -p logs runs

# The same thread settings as the array tasks; the dry run resolves the config and loads the
# objective without writing a run, and the env test checks the installed stack against the pins.
export OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
python -m experiments.run_bo --family aligned10 --seed 0 --cell product/lengthscale --T 200 \
    --out runs/ --objective-dir data/objectives --dry-run
python -m pytest tests/test_sagp_env.py -q
echo "setup complete: $REPO_DIR at $(git rev-parse --short HEAD), environment $ENV_NAME"
