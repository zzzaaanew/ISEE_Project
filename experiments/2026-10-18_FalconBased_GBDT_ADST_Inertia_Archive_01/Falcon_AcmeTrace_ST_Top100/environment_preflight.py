"""Small sequential backend fits; never modify the existing E19 environment."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def one_fit(name):
    import numpy as np
    rng = np.random.default_rng(20260905)
    x = rng.normal(size=(512, 16)).astype("float32")
    y = (x[:, 0] + 0.4*x[:, 1] > 0).astype("int32")
    start = time.perf_counter()
    if name == "xgboost_cuda":
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=5, max_depth=3, max_bin=32,
                                 tree_method="hist", device="cuda:0",
                                 n_jobs=2, random_state=20260905)
        model.fit(x, y)
        # Prediction via Booster avoids mismatched NumPy/CPU inplace-predict warnings.
        pred = model.get_booster().predict(xgb.DMatrix(x))
        detail = {"config": json.loads(model.get_booster().save_config()),
                  "build": xgb.build_info()}
    elif name == "catboost_gpu":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=5, depth=3, task_type="GPU", devices="0",
                                   gpu_ram_part=0.1, border_count=32, thread_count=2,
                                   random_seed=20260905, verbose=False,
                                   allow_writing_files=False)
        model.fit(x, y)
        pred = model.predict_proba(x)[:, 1]
        detail = {"parameters": model.get_all_params()}
    elif name in ("lightgbm_opencl", "lightgbm_cpu"):
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=5, max_depth=3, num_leaves=7,
                                   max_bin=31, device_type="gpu" if name.endswith("opencl") else "cpu",
                                   n_jobs=2, random_state=20260905, verbosity=1)
        model.fit(x, y)
        pred = model.predict_proba(x)[:, 1]
        detail = {"parameters": model.get_params()}
    elif name == "random_forest_cpu":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, max_depth=3, n_jobs=2,
                                        random_state=20260905)
        model.fit(x, y)
        pred = model.predict_proba(x)[:, 1]
        detail = {"parameters": model.get_params()}
    elif name == "isolation_forest_cpu":
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=5, max_samples=128, n_jobs=2,
                                 random_state=20260905)
        model.fit(x)
        pred = -model.score_samples(x)
        detail = {"parameters": model.get_params(), "direction": "negative score_samples: larger risk"}
    else:
        raise ValueError(name)
    assert np.isfinite(pred).all() and len(pred) == len(x)
    return {"status": "FIT_AND_PREDICT_CONFIRMED", "seconds": time.perf_counter()-start,
            "rows": len(x), "features": x.shape[1], "detail": detail}


def gpu_state():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return proc.stdout.strip() or proc.stderr.strip()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        try:
            result = one_fit(name)
        except Exception:
            result = {"status": "FAILED", "exception": traceback.format_exc()}
        (ROOT/"versions"/f"{name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"backend": name, "status": result["status"]}), flush=True)
        sys.exit(0 if result["status"] != "FAILED" else 1)
    (ROOT/"versions").mkdir(exist_ok=True)
    packages = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "catboost", "lightgbm",
                "psutil", "threadpoolctl", "scipy", "joblib"]
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {p: metadata.version(p) for p in packages}, "gpu_before": gpu_state(),
              "seed": 20260905, "synthetic_preflight_only": True,
              "existing_environments_modified": False, "fits": {},
              "backend_timeout_seconds": 120}
    log_path = ROOT/"versions"/"environment_preflight.log"
    for name in ["xgboost_cuda", "catboost_gpu", "lightgbm_opencl", "lightgbm_cpu",
                 "random_forest_cpu", "isolation_forest_cpu"]:
        print("Starting " + name, flush=True)
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
                                  capture_output=True, text=True, timeout=120,
                                  cwd=ROOT, encoding="utf-8", errors="replace")
            saved = ROOT/"versions"/f"{name}.json"
            fit = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {"status": "PROCESS_FAILURE"}
            fit.update({"returncode": proc.returncode, "wall_seconds": time.perf_counter()-start,
                        "stdout": proc.stdout, "stderr": proc.stderr})
        except subprocess.TimeoutExpired as exc:
            fit = {"status": "HARD_TIMEOUT", "wall_seconds": time.perf_counter()-start,
                   "stdout": str(exc.stdout), "stderr": str(exc.stderr)}
        result["fits"][name] = fit
        print(name + ": " + fit["status"], flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(name + ": " + fit["status"] + "\n" + fit.get("stdout", "") + fit.get("stderr", "") + "\n")
        (ROOT/"versions"/"environment_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["gpu_after"] = gpu_state()
    result["limitations"] = ["Tiny synthetic fits confirm backend capability, not full-data VRAM or speed.",
                              "GPU remains shared with desktop processes; no external process was terminated.",
                              "CPU-only sklearn families are retained as original algorithms."]
    (ROOT/"versions"/"environment_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
