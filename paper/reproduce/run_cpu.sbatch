#!/bin/bash
#SBATCH --job-name=harmonica_v2_cpu
#SBATCH --account=pi-ekempton
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/run_cpu_%j.log
# Run from paper/reproduce. CPU-only checks.
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=${PYTHON:-python}
for script in derivatives power2_precision quadrature_accuracy brute_force_areas; do
    $PY -u $script.py
done
