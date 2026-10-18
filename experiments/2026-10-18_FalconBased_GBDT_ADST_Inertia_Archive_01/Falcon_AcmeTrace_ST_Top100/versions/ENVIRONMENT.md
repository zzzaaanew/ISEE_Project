# Falcon backend preflight

Python executable: `../.venv/Scripts/python.exe` (CPython 3.12.14).

The existing E14/E19 environments were read for version information and were not modified.

| Algorithm | Confirmed backend | Small fit seconds |
|---|---|---:|
| XGBoost 2.1.4 | CUDA, `device=cuda:0`, `grow_gpu_hist` | 20.413 |
| CatBoost 1.2.8 | GPU, `task_type=GPU` | 2.169 |
| LightGBM 4.6.0 | OpenCL GPU, GTX 1060 3GB named in log | 9.430 |
| LightGBM 4.6.0 | CPU | 2.729 |
| Random Forest (sklearn 1.6.1) | CPU | 3.192 |
| Isolation Forest (sklearn 1.6.1) | CPU | 2.525 |

Evidence: `environment_preflight.json`, `environment_preflight.log`, per-backend JSON files.
Exact installed versions: `requirements-lock.txt`. `pip check` returned no broken requirements.

Each backend fit used only 512 synthetic rows, 16 features, and 5 trees (seed 20260905).
This establishes backend availability, not full-dataset GPU memory capacity, ML performance, or speed.
GPU was shared with desktop applications: 2810/3072 MiB and 99% utilization before the sequence,
2722/3072 MiB and 100% after. No external process was terminated.

XGBoost was pinned to 2.1.4 rather than a newer CUDA requirement incompatible with the local driver.
PyTorch was not installed. This is a separately pinned numerical environment; equivalence with E19's
NumPy/pandas/sklearn versions is not claimed.

LightGBM emitted OpenCL compile warnings while successfully fitting on the NVIDIA device.
LightGBM sklearn prediction emitted feature-name warnings for the synthetic NumPy input.
Neither warning caused fallback to CPU in the OpenCL fit.
