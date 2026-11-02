"""Execute selected, frozen legacy symbols without their import-time side effects.

Feature math, inertia, sample selection and ADST state transition are the original
AST nodes. The adapter changes scheduling / persistence, not those definitions.
"""
from __future__ import annotations

import ast
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
STEP_NS = 300_000_000_000
DAY_NS = STEP_NS * 288
METRICS = ["util", "temp", "power", "fb"]


def extract(path, names, env):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = []
    for node in tree.body:
        if getattr(node, "name", None) in names:
            selected.append(node)
        if isinstance(node, ast.ClassDef):
            selected.extend(x for x in node.body if getattr(x, "name", None) in names)
    assert len(selected) == len(names), (path, names)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future_annotations()), env)
    return env


def __future_annotations():
    import __future__
    return __future__.annotations.compiler_flag


base = SimpleNamespace(METRICS=METRICS, L_OBS_BINS_MAP={1:12,6:72,24:288},
                       MINUTE_NS=60_000_000_000, DAY_NS=DAY_NS, STEP_NS=STEP_NS)
env = dict(np=np, pd=pd, time=time, warnings=warnings, maximum_filter1d=maximum_filter1d,
           minimum_filter1d=minimum_filter1d, expit=expit, logit=logit, EPS=1e-6,
           average_precision_score=average_precision_score, runner=SimpleNamespace(base=base))
extract(ROOT / "sources/oct26_benchmark.py",
        {"_safe_ratio", "FastGridFeatureStore", "apply_kl_inertia_chain"}, env)
FastGridFeatureStore = env["FastGridFeatureStore"]
apply_kl_inertia_chain = env["apply_kl_inertia_chain"]

sample_env = dict(np=np)
extract(ROOT / "sources/run_bidirectional_adst_fusion.py",
        {"sample_indices", "adjusted_probability"}, sample_env)
sample_indices = sample_env["sample_indices"]
adjusted_probability = sample_env["adjusted_probability"]

controller_env = dict(np=np, TRAIN_WINDOW_CANDIDATES=(7,14,21),
                      ADST_DROP_THRESHOLD=.15, ADST_STABLE_ORIGINS=3)
extract(ROOT / "sources/run_experiment11_30m_han_adst_b1_light_cascade_b2_sliding.py",
        {"_safe_norm_ap", "_safe_mean", "_update_adst_from_cycle"}, controller_env)
update_adst = controller_env["_update_adst_from_cycle"]


def new_controller():
    return SimpleNamespace(current_train_days=14, adst_stable_count=0,
                           adst_history=[], pending_adst_action="initial14d")


def basic30(store, bins, gpus, hours):
    roll = store.get_rolling(hours)
    cols = [roll[f"{m}_{s}"][bins,gpus] for m in METRICS
            for s in ["last","mean","max","min","std","delta"]]
    nodes = gpus // 8
    node_vals = [np.nan_to_num(store.engine.node_matrices[m][bins-3,nodes], nan=0.)
                 for m in ["util","temp","power"]]
    cols += node_vals
    cols += [roll[f"{m}_last"][bins,gpus]-v for m,v in zip(["util","temp","power"],node_vals)]
    return np.nan_to_num(np.column_stack(cols), nan=0., posinf=0., neginf=0.).astype(np.float32)


def feature_names():
    common = [f"{m}_{s}" for m in METRICS for s in ["last","mean","std","delta"]]
    context = [x for m in ["util","temp","power"] for x in [f"{m}_node_mean",f"{m}_diff_node"]]
    basic = [f"{m}_{s}" for m in METRICS for s in ["last","mean","max","min","std","delta"]]
    basic += [f"{m}_node_mean" for m in ["util","temp","power"]]
    basic += [f"{m}_diff_node" for m in ["util","temp","power"]]
    cross = ["thermal_efficiency","power_temp_ratio","util_fb_coupling","power_per_util",
             "temp_fb_divergence","thermal_headroom","power_headroom"]
    spatial = ["gnn_neighbor_temp_max","gnn_neighbor_temp_mean","gnn_node_temp_std","gnn_temp_spatial_diff",
               "gnn_neighbor_power_max","gnn_neighbor_power_mean","gnn_power_spatial_diff","gnn_neighbor_util_mean"]
    extra = ["peer_temp_zscore","peer_power_zscore","peer_util_zscore","peer_temp_rank","peer_power_rank",
             "temp_slope","power_slope","thermal_persistence"]
    return dict(basic30=basic, falcon45=common+context+cross+spatial+extra)
