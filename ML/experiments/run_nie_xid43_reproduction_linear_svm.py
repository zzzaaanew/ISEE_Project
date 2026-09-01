"""Run the Nie XID 43 reproduction with a tractable linear SVM.

The paper used an RBF SVM, but exact RBF inference over the current project's
millions of eligible GPU-time test samples is prohibitively expensive.  This
wrapper preserves the same feature construction, target, TwoStage gate, and
other three models while replacing only the SVM kernel with LinearSVC.  The
model is reported as ``linear_svm`` so the approximation is explicit.
"""

from __future__ import annotations

import numpy as np

import run_nie_xid43_reproduction as experiment

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def fit_models_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    svm_max_samples: int,
) -> tuple[dict, StandardScaler, dict]:
    del svm_max_samples
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train).astype(np.float32, copy=False)
    models = {}
    train_info = {}

    models["logistic"] = LogisticRegression(
        solver="lbfgs", max_iter=500, C=1.0, random_state=seed
    ).fit(x_scaled, y_train)
    train_info["logistic_samples"] = int(len(y_train))

    models["gbdt"] = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed + 1,
    ).fit(x_train, y_train)
    train_info["gbdt_samples"] = int(len(y_train))

    models["linear_svm"] = LinearSVC(
        C=1.0, class_weight=None, max_iter=5000, dual="auto", random_state=seed + 2
    ).fit(x_scaled, y_train)
    train_info["linear_svm_samples"] = int(len(y_train))

    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=seed + 3,
    ).fit(x_scaled, y_train)
    train_info["mlp_samples"] = int(len(y_train))
    return models, scaler, train_info


def model_scores_linear_svm(models: dict, scaler: StandardScaler, x: np.ndarray) -> dict:
    x_scaled = scaler.transform(x).astype(np.float32, copy=False)
    decision = models["linear_svm"].decision_function(x_scaled)
    return {
        "logistic": models["logistic"].predict_proba(x_scaled)[:, 1].astype(np.float32),
        "gbdt": models["gbdt"].predict_proba(x)[:, 1].astype(np.float32),
        "linear_svm": experiment.sigmoid(decision),
        "mlp": models["mlp"].predict_proba(x_scaled)[:, 1].astype(np.float32),
    }


experiment.fit_models = fit_models_linear_svm
experiment.model_scores = model_scores_linear_svm


if __name__ == "__main__":
    experiment.main()
