from pathlib import Path
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "eda_extended_metrics"
MATCH_DIR = ROOT / "outputs" / "eda_matched_controls"


def bootstrap(values, statistic, repeats=5000):
    rng = np.random.default_rng(45)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True)
    estimates = statistic(draws, axis=1)
    return np.quantile(estimates, [0.025, 0.975])


def selected_pairs():
    matched = pd.read_csv(MATCH_DIR / "matched_pairs.csv")
    return matched[(matched.event_activity == "high") & matched.xid_code.isin([31, 43])].sort_values("match_id").reset_index(drop=True)


def thermal_test(pairs):
    values = np.load(OUTPUT / "memory_temp_paired_curves.npz")["values"]
    size = len(pairs)
    rows = []
    for code in [31, 43]:
        mask = pairs.xid_code.eq(code).to_numpy()
        frame = pairs.loc[mask, ["cluster_id"]].reset_index(drop=True)
        for source, curves in [("event", values[:size][mask]), ("control", values[size:][mask])]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline = np.nanmedian(curves[:, :30], axis=1)
                peak = np.nanmax(curves[:, 50:60], axis=1)
            valid = np.isfinite(curves[:, :30]).sum(axis=1) >= 15
            frame[f"{source}_rise"] = np.where(valid, peak - baseline, np.nan)
        clusters = frame.groupby("cluster_id").median().dropna()
        difference = (clusters.event_rise - clusters.control_rise).to_numpy()
        risk_difference = (clusters.event_rise.ge(5).astype(float) - clusters.control_rise.ge(5).astype(float)).to_numpy()
        median_ci = bootstrap(difference, np.median)
        risk_ci = bootstrap(risk_difference, np.mean)
        rows.append(
            {
                "xid_code": code,
                "node_event_clusters": len(clusters),
                "median_event_peak_rise_c": clusters.event_rise.median(),
                "median_control_peak_rise_c": clusters.control_rise.median(),
                "median_paired_difference_c": np.median(difference),
                "median_difference_ci_low": median_ci[0],
                "median_difference_ci_high": median_ci[1],
                "event_rise_ge5_fraction": clusters.event_rise.ge(5).mean(),
                "control_rise_ge5_fraction": clusters.control_rise.ge(5).mean(),
                "paired_risk_difference": risk_difference.mean(),
                "risk_difference_ci_low": risk_ci[0],
                "risk_difference_ci_high": risk_ci[1],
            }
        )
    return pd.DataFrame(rows)


def sensitivity(pair_mechanism):
    rows = []
    for code in [31, 43]:
        clusters = pair_mechanism[pair_mechanism.xid_code.eq(code)].groupby("cluster_id").median(numeric_only=True)
        for drop_pct in [10, 20, 30, 50]:
            threshold = 1 - drop_pct / 100
            valid = clusters[["event_node_power_ratio", "control_node_power_ratio"]].notna().all(axis=1)
            event = clusters.event_node_power_ratio.le(threshold)
            control = clusters.control_node_power_ratio.le(threshold)
            for composite in [False, True]:
                selected = valid.copy()
                if composite:
                    selected &= clusters[["event_terminal_low_minutes_u10", "control_terminal_low_minutes_u10"]].notna().all(axis=1)
                    event_flag = event & clusters.event_terminal_low_minutes_u10.ge(3)
                    control_flag = control & clusters.control_terminal_low_minutes_u10.ge(3)
                else:
                    event_flag, control_flag = event, control
                difference = (event_flag[selected].astype(float) - control_flag[selected].astype(float)).to_numpy()
                ci = bootstrap(difference, np.mean)
                rows.append(
                    {
                        "xid_code": code,
                        "definition": "gpu_low_and_node_power_drop" if composite else "node_power_drop",
                        "power_drop_threshold_pct": drop_pct,
                        "node_event_clusters": int(selected.sum()),
                        "event_fraction": event_flag[selected].mean(),
                        "control_fraction": control_flag[selected].mean(),
                        "paired_risk_difference": difference.mean(),
                        "bootstrap_ci_low": ci[0],
                        "bootstrap_ci_high": ci[1],
                    }
                )
    return pd.DataFrame(rows)


def main():
    pairs = selected_pairs()
    mechanisms = pd.read_csv(OUTPUT / "extended_pair_mechanisms.csv")
    classes = pd.read_csv(OUTPUT / "xid43_event_classification.csv")
    effects = pd.read_csv(OUTPUT / "extended_effect_summary.csv")
    thermal = thermal_test(pairs)
    power = sensitivity(mechanisms)
    class_summary = classes.event_type.value_counts().rename_axis("event_type").reset_index(name="node_event_clusters")
    class_summary["fraction_all_xid43_high"] = class_summary.node_event_clusters / len(classes)
    thermal.to_csv(OUTPUT / "thermal_peak_test.csv", index=False)
    power.to_csv(OUTPUT / "node_power_sensitivity.csv", index=False)
    class_summary.to_csv(OUTPUT / "xid43_event_classification_summary.csv", index=False)

    t43 = thermal[thermal.xid_code.eq(43)].iloc[0]
    p43 = power[(power.xid_code.eq(43)) & (power.definition == "node_power_drop") & power.power_drop_threshold_pct.eq(20)].iloc[0]
    pc43 = power[(power.xid_code.eq(43)) & (power.definition == "gpu_low_and_node_power_drop") & power.power_drop_threshold_pct.eq(20)].iloc[0]
    e43 = effects[(effects.xid_code.eq(43)) & (effects.metric == "memory_temp") & (effects.comparison == "pre_5_0m")].iloc[0]
    n43 = effects[(effects.xid_code.eq(43)) & (effects.metric == "node_total_power") & (effects.comparison == "pre_10_0m")].iloc[0]
    clock = effects[(effects.xid_code.eq(43)) & (effects.metric == "mem_clock") & (effects.comparison == "pre_5_0m")].iloc[0]
    node_like = int((classes.event_type == "node_or_workload_wide_shutdown_like").sum())
    gpu_like = int((classes.event_type == "gpu_specific_shutdown_like").sum())
    classifiable_shutdown = node_like + gpu_like

    lines = [
        "# MEMORY_TEMP·MEM_CLOCK·IPMI 확장 EDA 핵심 결과",
        "",
        "## 분석 설계",
        "",
        "- 기존 동일 GPU·요일·시간대·high-activity matched control pair를 그대로 사용했다.",
        "- GPU metric은 -60~+60분 1분 bin, IPMI 전력은 같은 구간의 5분 bin으로 정렬했다.",
        "- 동일 node-time 사건의 여러 GPU는 먼저 cluster 중앙값으로 집계했다.",
        "- 사건 후 값은 현상 설명용이며 예측 feature로 사용하지 않는다.",
        "",
        "## 1. HBM 과열 가설: 지지되지 않음",
        "",
        f"- XID 43의 사건 전 5분 MEMORY_TEMP 변화는 control 대비 {e43.median_difference_in_change:.2f}°C였다 (bootstrap 95% CI {e43.bootstrap_ci_low:.2f}~{e43.bootstrap_ci_high:.2f}).",
        f"- 사건 전 10분 최고온도의 baseline 대비 변화 중앙값은 사건 {t43.median_event_peak_rise_c:.2f}°C, control {t43.median_control_peak_rise_c:.2f}°C였다.",
        f"- 5°C 이상 상승 비율은 사건 {t43.event_rise_ge5_fraction:.1%}, control {t43.control_rise_ge5_fraction:.1%}였다; paired 차이의 95% CI는 {t43.risk_difference_ci_low:+.1%}~{t43.risk_difference_ci_high:+.1%}다.",
        "- 따라서 XID 43 직전에는 HBM 발열 상승이 아니라 workload 정지에 따른 냉각이 관찰된다.",
        "",
        "## 2. Memory clock 선행 붕괴 가설: 지지되지 않음",
        "",
        f"- 사건과 control 양쪽이 관측된 XID 43 표본은 {int(clock.gpu_pairs)} GPU pair, {int(clock.node_event_clusters)} cluster다.",
        f"- 사건 전 5분의 matched 변화 차이는 {clock.median_difference_in_change:.1f} MHz였고 CI도 {clock.bootstrap_ci_low:.1f}~{clock.bootstrap_ci_high:.1f} MHz였다.",
        "- 유효 cluster에서 마지막 3분 memory clock/baseline 비율은 사건과 control 모두 정확히 1.0이었으며 50% 이하 붕괴는 0건이었다.",
        "- 이 metric은 XID 시각 판별이나 조기경보 feature로 유용하지 않다. 고정 P-state 또는 last-known 값일 가능성이 있다.",
        "",
        "## 3. 노드 전력 하락: 일부 사건에서 명확",
        "",
        f"- XID 43 직전 10분 node total power 변화는 사건 {n43.median_event_change:.1f} W, control {n43.median_control_change:.1f} W였다.",
        f"- matched difference-in-change는 {n43.median_difference_in_change:.1f} W (95% CI {n43.bootstrap_ci_low:.1f}~{n43.bootstrap_ci_high:.1f})였다.",
        f"- baseline 대비 20% 이상 전력 하락: 사건 {p43.event_fraction:.1%}, control {p43.control_fraction:.1%}; paired 차이 {p43.paired_risk_difference:+.1%} (CI {p43.bootstrap_ci_low:+.1%}~{p43.bootstrap_ci_high:+.1%}).",
        f"- GPU가 3분 이상 비활성이면서 node power도 20% 이상 하락: 사건 {pc43.event_fraction:.1%}, control {pc43.control_fraction:.1%}; paired 차이 {pc43.paired_risk_difference:+.1%} (CI {pc43.bootstrap_ci_low:+.1%}~{pc43.bootstrap_ci_high:+.1%}).",
        "",
        "## 4. XID 43 사건 유형",
        "",
        class_summary.to_markdown(index=False),
        "",
        f"- GPU 사전 비활성과 IPMI가 모두 관측된 shutdown-like cluster {classifiable_shutdown}개 중 node/workload-wide 유형은 {node_like}/{classifiable_shutdown} ({node_like/classifiable_shutdown:.1%}), GPU-specific 유형은 {gpu_like}/{classifiable_shutdown} ({gpu_like/classifiable_shutdown:.1%})였다.",
        "- node/workload-wide 유형의 node power 비율 중앙값은 baseline의 약 56%였고 완전한 0에 가깝지는 않았다. 따라서 물리적 node power-off보다 workload 종료, GPU subsystem 정지 또는 reset 가능성이 더 높다.",
        "",
        "## 파이프라인 반영",
        "",
        "1. MEMORY_TEMP 상승과 MEM_CLOCK 붕괴는 XID 43 조기예측 feature 우선순위에서 제외한다.",
        "2. node total power는 사건 유형 분리용 context feature로 유지한다.",
        "3. XID 43을 하나의 label로만 학습하지 말고 GPU-specific과 node/workload-wide 하위 유형을 구분한다.",
        "4. GPU_UTIL 저하와 node power 저하는 이미 기능적 정지 이후일 수 있으므로 early-warning 모델에서는 ambiguous/leakage zone으로 처리한다.",
        "5. IPMI는 5분 간격이므로 정확한 장애 시각 추정보다는 유형 분류에 사용한다.",
        "",
        "## 제한",
        "",
        "- MEM_CLOCK은 7월 이후에만 존재해 paired 표본이 작다.",
        "- IPMI baseline 절대 수준은 사건/control에서 다를 수 있어 절대 전력보다 각 window의 baseline 대비 변화량을 비교했다.",
        "- node power 하락만으로 정상 job 종료와 node/GPU reset을 구분할 수 없다.",
    ]
    (OUTPUT / "EXTENDED_METRICS_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[done] postprocess")


if __name__ == "__main__":
    main()
