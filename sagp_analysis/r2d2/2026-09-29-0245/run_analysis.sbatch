#!/bin/bash
#SBATCH --job-name=r2d2-analysis
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=%x-%j.out
# The R2-D2 study analysis (plan D11, gate G3). Pilot (G3):
#   sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b run_analysis.sbatch
# Stage 4 (all eight families, ten seeds; plus-separated because Slurm splits --export on commas):
#   sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b,SEEDS=0-9,FAMILIES=aligned3+aligned10+anti_aligned+decoupled+interaction_g0.00+interaction_g0.25+interaction_g0.50+interaction_g0.75 run_analysis.sbatch
set -euo pipefail
A="${SLURM_SUBMIT_DIR:?run via sbatch from the analysis dir}"
cd "$A"
REPO="${REPO:-$WRKDIR/saasbo}"
echo "node: $(hostname)  start: $(date -Is)  repo: $REPO at $(git -C "$REPO" rev-parse --short HEAD)"
if [ ! -x "$A/venv/bin/python" ]; then
  echo "building venv under $A/venv"
  module load mamba
  source activate saasbo
  python -m venv "$A/venv"
  conda deactivate || true
  "$A/venv/bin/python" -m pip install --quiet --upgrade pip
  "$A/venv/bin/python" -m pip install --quiet numpy scipy pandas matplotlib
  "$A/venv/bin/python" -m pip freeze > "$A/venv-freeze.txt"
fi
"$A/venv/bin/python" -c "import numpy, scipy, pandas, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'matplotlib', matplotlib.__version__)"
"$A/venv/bin/python" "$A/analyze.py" --runs "$REPO/runs" --runs "$REPO/runs_r2d2" --repo "$REPO" --out "$A" \
  --asof "${ASOF:-}" --launch-commit "${LAUNCH_COMMIT:?set LAUNCH_COMMIT, the R2-D2 launch commit (97eda925b)}" --families "$(echo "${FAMILIES:-aligned10+decoupled}" | tr "+" ",")" --seeds "${SEEDS:-0-4}"
echo "end: $(date -Is)"
