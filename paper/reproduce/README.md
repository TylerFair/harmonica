# Reproducing the paper's numbers

Every script imports the installed `harmonica`, so check out and install the
code you want to test (`pip install -e .` from the repository root), then run
from this directory. Each script writes a JSON file to `results/`, recording
the harmonica commit, jax version and devices it ran with.

| Script | Paper | What it checks |
|---|---|---|
| `brute_force_areas.py` | Section 4, overlap areas | 6000 random strings against brute-force overlap areas; flagged cases recomputed at 40x the resolution |
| `quadrature_accuracy.py` | Table 1, C++ parity | Flux error against 500-point C++ quadrature, by quadrature order, law and N_c |
| `power2_precision.py` | Section 3.4 | The power-2 kernel and its first two derivatives against a 75-digit reference |
| `derivatives.py` | Section 4, derivatives and orbits | Forward vs reverse mode, contact derivatives, the C++ path's Hessians, Kepler |
| `gpu_benchmark.py` | Table 2, Section 5 | Timings against the C++ on one core of the same node, including the order-six case |
| `tutorial_nuts.py` | Section 4, inference | harmonica's NUTS tutorial (N_c = 3), with the posterior and run time |

`run_cpu.sbatch` runs the first four on a CPU node and `run_gpu.sbatch` the last
two on a GPU (set `PYTHON` to your interpreter). GPU timings depend on the card
and on whether it is shared; run on an otherwise idle GPU for meaningful numbers.
