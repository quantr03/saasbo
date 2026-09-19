#!/bin/bash
#SBATCH --job-name=sagp-narrative2
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/logs/compute2-%j.out
set -euo pipefail
echo "node: $(hostname) start: $(date -Is)"
/scratch/work/tranq8/sagp_analysis/latest/venv/bin/python /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/compute2.py /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849
echo "end: $(date -Is)"
