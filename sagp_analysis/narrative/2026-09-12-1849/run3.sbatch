#!/bin/bash
#SBATCH --job-name=sagp-narrative-html
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/logs/html-%j.out
set -euo pipefail
/scratch/work/tranq8/sagp_analysis/latest/venv/bin/python /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849/build_html.py /scratch/work/tranq8/sagp_analysis/narrative/2026-09-12-1849
