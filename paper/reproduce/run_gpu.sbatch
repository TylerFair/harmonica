#!/bin/bash
#SBATCH --job-name=harmonica_v2_gpu
#SBATCH --account=pi-ekempton
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/run_gpu_%j.log
# Run from paper/reproduce, ideally on an otherwise idle GPU.
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PYTHON:-python}
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
$PY -u gpu_benchmark.py
$PY -u tutorial_nuts.py
