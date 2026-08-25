from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_risk_tape_5m as pipeline


def finalize() -> None:
    output = pipeline.OUTPUT
    metrics = pd.read_csv(output / "fold_metrics.csv")
    sample = pd.read_parquet(
        output / "decision_sample.parquet",
        columns=["sample_id", "label_xid31_or_43_within_5m"],
    )
    with np.load(output / "risk_tape_5m.npz") as tape:
        rank = tape["risk_rank"]
        fold_id = tape["model_fold_id"]
    positives = sample.loc[sample["label_xid31_or_43_within_5m"].eq(1), "sample_id"].to_numpy()
    bins, gpus = positives // 1_992, positives % 1_992
    keep = fold_id[bins] > 0
    positive_ranks = rank[bins[keep], gpus[keep]]
    overall = {f"overall_recall_at_{k}": float((positive_ranks <= k).mean()) for k in [10, 50, 100]}

    meta_path = output / "risk_tape_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "model_fold_id_map": {"-1": "warmup_no_oof", "1": "fold_1", "2": "fold_2", "3": "fold_3", "4": "fold_4"},
            "overall_oof_positive_gpu_bins": int(keep.sum()),
            **overall,
            "full_matrix_contract": {
                "risk_score": "float32; NaN in warmup",
                "risk_rank": "uint16; 0 in warmup, otherwise 1..1992 per decision_time",
                "observability_score": "float16; 0..1, lower means less observable",
            },
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_path = output / "RISK_TAPE_5M_FINDINGS.md"
    report = report_path.read_text(encoding="utf-8").split("\n## 최종 판정", 1)[0].rstrip()
    report += (
        "\n\n## 최종 판정\n\n"
        f"- 전체 OOF positive {int(keep.sum()):,}건 중 top-10 recall {overall['overall_recall_at_10']:.1%}, "
        f"top-50 {overall['overall_recall_at_50']:.1%}, top-100 {overall['overall_recall_at_100']:.1%}.\n"
        f"- fold별 ROC-AUC 범위 {metrics.roc_auc.min():.3f}~{metrics.roc_auc.max():.3f}, "
        f"top-100 recall 범위 {metrics.recall_at_100.min():.1%}~{metrics.recall_at_100.max():.1%}.\n"
        "- 무작위 순위보다 우수한 신호는 있으나 기간별 변동성이 크다. 현재 Risk Tape는 Blox 정책 비교용 탐색적 baseline으로 사용하고 안정된 운영 예측기로 주장하지 않는다.\n"
    )
    report_path.write_text(report, encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(metrics))
    axes[0].bar(x, metrics["roc_auc"], color="#2878B5", label="ROC-AUC")
    axes[0].set_ylim(0.5, 1)
    axes[0].set_xticks(x, metrics["model_fold"])
    axes[0].set_title("OOF discrimination")
    secondary = axes[0].twinx()
    secondary.plot(x, metrics["average_precision"] / metrics["prevalence"], color="#E87500", marker="o", label="AP / prevalence")
    secondary.set_yscale("log")
    secondary.set_ylabel("AP lift (log scale)")
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = secondary.get_legend_handles_labels()
    axes[0].legend(lines + lines2, labels + labels2, loc="upper right")
    for key, label in [("recall_at_10", "K=10"), ("recall_at_50", "K=50"), ("recall_at_100", "K=100")]:
        axes[1].plot(x, metrics[key], marker="o", label=label)
    axes[1].set_xticks(x, metrics["model_fold"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Positive recall among top-K GPUs")
    axes[1].legend()
    for key, label in [("lift_at_10", "K=10"), ("lift_at_50", "K=50"), ("lift_at_100", "K=100")]:
        axes[2].plot(x, metrics[key], marker="o", label=label)
    axes[2].set_xticks(x, metrics["model_fold"])
    axes[2].set_yscale("log")
    axes[2].set_title("Top-K lift over prevalence")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "risk_oof_performance.png", dpi=160)
    plt.close(fig)
    assert meta["model_fold_id_map"]["-1"] == "warmup_no_oof"
    print(json.dumps({**overall, "oof_positives": int(keep.sum())}, indent=2))


if __name__ == "__main__":
    if "--rescan" in sys.argv:
        pipeline.main()
    finalize()
