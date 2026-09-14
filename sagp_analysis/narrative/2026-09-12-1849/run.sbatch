#!/bin/bash
#SBATCH --job-name=sagp-narrative
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/logs/compute-%j.out
set -euo pipefail
echo "node: $(hostname) start: $(date -Is)"
PY=/scratch/work/tranq8/sagp_analysis/latest/venv/bin/python
$PY /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/compute_eps.py /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849
$PY /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/compute.py /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849
echo "end: $(date -Is)"
