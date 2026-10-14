# Discrete–continuum dislocation switching

Code, saved results and figure scripts for the manuscript *Error-controlled switching between discrete and continuum dislocations in a unified defect framework* (A. Linda, Indian Institute of Technology Kanpur). The model and its derivations are described in the paper; a link to the preprint will be added here when it is available.

## Installation and tests

Python 3.11 or newer.

~~~sh
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"                      # numpy, scipy, matplotlib, numba, pytest
python -m pytest -q                                     # 363 tests; the ASE test is skipped unless ".[atomistics]" is installed
~~~

Optional extras: `.[atomistics]` (ASE, for the EMT/EAM molecular statics), `.[gpu]` (CuPy, 3D creep), `.[elastic-line]` (PyTorch, elastic Frank–Read sources) and `.[benchmarks]` (psutil). The 3D allocator test uses [ExaDiS/OpenDiS](https://github.com/OpenDiS/OpenDiS) built from source; the driver is `tools/exadis/vein_channel.py`.

## Reproducing the results

Run the benchmark scripts from the repository root with one numerical thread, for example

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/representation_allocator.py
~~~

Each run directory in `results/` stores a `report.json` with the seeds, parameters, software versions and platform of the run. Home-directory paths and host names in the saved reports were anonymised; the numbers are unchanged.

| Paper result | Saved data (`results/`) | Script (`benchmarks/`) |
| --- | --- | --- |
| Interface-focused vs uniform grids | `performance-complete-2026-09-13`, `graded-cost-2026-09-26` | `performance_audit.py`, `graded_cost.py` |
| AMR baseline | `amr-message-passing-2026-09-26/amr` | `amr_baseline.py` |
| Crossover map and parallel scaling | `crossover-map-2026-09-27`, `parallel-scaling-2026-09-26`, `amr-message-passing-2026-09-26/message-passing` | `crossover_map.py`, `parallel_scaling.py`, `message_passing.py` |
| Discrete–continuum allocator and error control | `representation-allocator-2026-09-28`, `allocator-control-2026-09-28` | `representation_allocator.py`, `representation_allocator_control.py`, `representation_allocator_timing.py` |
| Allocator validation (second machine) | `allocator-validation-2026-09-30` | `allocator_validation.py` |
| Copper wall array and equal-cost comparison | `realistic-copper-2026-09-30`, `realistic-pareto-2026-09-30`, `pfdd-estimate-2026-09-30` | `realistic_copper.py`, `realistic_pareto.py`, `pfdd_estimate.py` |
| 3D allocator test with ExaDiS | `allocator-3d-2026-09-30`, `allocator-3d-wall-2026-10-01` | `tools/exadis/vein_channel.py`, `allocator3d_report.py` |
| Creep cavitation (2D and 3D) | `creep-cavitation-2026-09-30`, `creep-cavitation-3d-2026-09-30` (v1 calibration: `creep-cavitation-3d-v1-2026-09-30`) | `creep_cavitation.py`, `creep_cavitation3d.py`, `creep_calibration3d.py` |
| Verification studies (Supporting Information) | the remaining dated folders in `results/` | the script of the same name in `benchmarks/` |

Figures are redrawn from the saved results by the scripts in `manuscript/figures/` (for example `make_schematic.py`, `make_figures.py`, `make_validation_figs.py`, `make_realistic_fig.py`, `make_pareto_fig.py`, `make_alloc3d_fig.py`, `make_creep_fig.py` and `make_si_figs.py`).

## Layout

- `src/bmvr/`: the Python package (fields, dislocation particles and densities, allocator, mechanics, fracture, transport).
- `benchmarks/`: drivers and report scripts for every study.
- `tests/`: unit and regression tests.
- `results/`: saved outputs of every run.
- `tools/exadis/`: driver and build scripts for the ExaDiS calculations.
- `data/potentials/`: interatomic potential used in the atomistic calculations.

## Licence and citation

Code (`src/`, `benchmarks/`, `tests/`, `tools/`, figure scripts): [MIT](LICENSE). Results and data (`results/`, `data/`): [CC BY 4.0](LICENSE-DATA). The interatomic potential in `data/potentials/` keeps the terms of its original source (NIST Interatomic Potentials Repository). To cite, see [CITATION.cff](CITATION.cff).
