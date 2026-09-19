#!/bin/bash
#SBATCH --job-name=sagp-analysis
#SBATCH --time=00:40:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=%x-%j.out
set -euo pipefail
A="${SLURM_SUBMIT_DIR:?run via sbatch from the analysis dir}"
cd "$A"
echo "node: $(hostname)  start: $(date -Is)"
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
"$A/venv/bin/python" "$A/analyze.py" --runs /scratch/work/tranq8/saasbo/runs --out "$A" --asof "${ASOF:-}"
echo "end: $(date -Is)"
