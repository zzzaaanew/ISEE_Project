from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent.parent / "outputs" / "eda_event_windows"
METRICS = ["dram_active", "fb_used", "gpu_temp", "gpu_util", "power_usage"]
OFFSETS = np.arange(-60, 60)


def active_effects(events, features, active_ids):
    selected = features[
        features["event_id"].isin(active_ids)
        & features["isolated_2h"]
        & features["xid_code"].isin([31, 43])
    ]
    rows = []
    for (code, metric), group in selected.groupby(["xid_code", "metric"]):
        for target in ["pre_5_0m", "post_0_5m"]:
            pairs = group[["cluster_id", "baseline_60_30m", target]].dropna()
            pairs = pairs.assign(delta=pairs[target] - pairs["baseline_60_30m"])
            clusters = pairs.groupby("cluster_id", as_index=False).median(numeric_only=True)
            rows.append(
                {
                    "xid_code": code,
                    "metric": metric,
                    "comparison": target,
                    "gpu_pairs": len(pairs),
                    "node_event_clusters": len(clusters),
                    "median_baseline": clusters["baseline_60_30m"].median(),
                    "median_target": clusters[target].median(),
                    "median_delta": clusters["delta"].median(),
                    "delta_q25": clusters["delta"].quantile(0.25),
                    "delta_q75": clusters["delta"].quantile(0.75),
                    "clusters_increased_fraction": (clusters["delta"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def active_curves(events, active_ids):
    rows = []
    active = events["event_id"].isin(active_ids).to_numpy()
    for metric in METRICS:
        values = np.load(OUT / f"{metric}_event_curves.npz")["short"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            baseline = np.nanmedian(values[:, :30], axis=1)
        normalized = values - baseline[:, None]
        normalized[np.isfinite(values[:, :30]).sum(axis=1) < 15] = np.nan
        for code in [31, 43]:
            mask = active & events["isolated_2h"].to_numpy() & events["xid_code"].eq(code).to_numpy()
            frame = pd.DataFrame(normalized[mask])
            frame.insert(0, "cluster_id", events.loc[mask, "cluster_id"].to_numpy())
            clusters = frame.groupby("cluster_id").median(numeric_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(clusters, axis=0)
                q25 = np.nanpercentile(clusters, 25, axis=0)
                q75 = np.nanpercentile(clusters, 75, axis=0)
            valid = np.isfinite(clusters).sum(axis=0).to_numpy()
            for index, offset in enumerate(OFFSETS):
                rows.append(
                    {
                        "metric": metric,
                        "xid_code": code,
                        "offset_min": offset,
                        "node_event_clusters": int(valid[index]),
                        "median_delta": median[index],
                        "q25": q25[index],
                        "q75": q75[index],
                    }
                )
    return pd.DataFrame(rows)


def plot(curves):
    figure, axes = plt.subplots(len(METRICS), 1, figsize=(12, 14), sharex=True)
    colors = {31: "tab:blue", 43: "tab:orange"}
    for axis, metric in zip(axes, METRICS, strict=True):
        for code in [31, 43]:
            data = curves[(curves["metric"] == metric) & (curves["xid_code"] == code)]
            x = data["offset_min"].to_numpy(dtype=float) + 0.5
            median = data["median_delta"].to_numpy(dtype=float)
            q25 = data["q25"].to_numpy(dtype=float)
            q75 = data["q75"].to_numpy(dtype=float)
            axis.plot(x, median, label=f"XID {code}", color=colors[code])
            axis.fill_between(x, q25, q75, color=colors[code], alpha=0.15)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set_ylabel(metric)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Minutes from episode start")
    figure.suptitle("Active GPUs before XID episodes — change from -60~-30 min baseline")
    figure.tight_layout()
    figure.savefig(OUT / "event_active_short_curves.png", dpi=160)
    plt.close(figure)


def write_findings(events, effects, curves, active_ids):
    counts = (
        events[events["event_id"].isin(active_ids) & events["isolated_2h"] & events["xid_code"].isin([31, 43])]
        .groupby("xid_code")
        .agg(gpu_episodes=("event_id", "nunique"), node_event_clusters=("cluster_id", "nunique"))
    )

    def point(code, metric, minute):
        row = curves[(curves["xid_code"] == code) & (curves["metric"] == metric) & (curves["offset_min"] == minute)].iloc[0]
        return row["median_delta"]

    table = effects[
        [
            "xid_code",
            "metric",
            "comparison",
            "node_event_clusters",
            "median_baseline",
            "median_target",
            "median_delta",
            "clusters_increased_fraction",
        ]
    ].copy()
    table["clusters_increased_fraction"] = table["clusters_increased_fraction"].map(lambda value: f"{value:.1%}")
    lines = [
        "# Active-state XID episode 전후 분석",
        "",
        "## 기준",
        "",
        "- 사건 전 -60~-30분 GPU_UTIL 중앙값이 10% 이상인 GPU episode만 active로 정의했다.",
        "- 전후 2시간 내 다른 episode가 없는 사건만 사용했다.",
        "- 동일 node·시각·XID code의 GPU는 먼저 cluster 중앙값으로 묶었다.",
        f"- XID 31: GPU episode {int(counts.loc[31, 'gpu_episodes']):,}개, node-event cluster {int(counts.loc[31, 'node_event_clusters']):,}개.",
        f"- XID 43: GPU episode {int(counts.loc[43, 'gpu_episodes']):,}개, node-event cluster {int(counts.loc[43, 'node_event_clusters']):,}개.",
        "",
        "## 핵심 결과",
        "",
        f"- XID 31은 t-5분의 GPU_UTIL 변화 중앙값이 {point(31, 'gpu_util', -5):.2f}%p이고 t=0에서 {point(31, 'gpu_util', 0):.2f}%p로 급락한다.",
        f"- XID 31은 t=0에서 POWER_USAGE {point(31, 'power_usage', 0):.2f}, GPU_TEMP {point(31, 'gpu_temp', 0):.2f}, FB_USED {point(31, 'fb_used', 0):.2f}만큼 기준선보다 낮아진다.",
        f"- XID 43은 t-10분부터 GPU_UTIL {point(43, 'gpu_util', -10):.2f}%p, POWER_USAGE {point(43, 'power_usage', -10):.2f}, GPU_TEMP {point(43, 'gpu_temp', -10):.2f}의 하락이 보인다.",
        f"- XID 43의 GPU_UTIL 변화는 t-2분 {point(43, 'gpu_util', -2):.2f}%p, t=0 {point(43, 'gpu_util', 0):.2f}%p다.",
        "- 두 코드 모두 사건 직후 사용률·전력·메모리·온도가 내려간다. 온도나 전력이 점진적으로 상승하는 전형적 과열 패턴은 중앙값 수준에서 확인되지 않았다.",
        "",
        "## Active 사건의 윈도우 효과",
        "",
        table.to_markdown(index=False),
        "",
        "## 파이프라인 시사점",
        "",
        "- XID 31과 43은 전조 형태가 달라 코드별 모델 또는 code-conditioned head가 필요하다.",
        "- XID 31의 5분 전 텔레메트리만으로는 중앙값 기준 뚜렷한 전조가 약하다.",
        "- XID 43의 사전 하락은 예측 신호 후보지만, 실제 고장보다 XID 기록이 늦거나 작업 종료·GPU reset을 반영할 수 있다.",
        "- t=0 이후 값은 결과 변수 누출이므로 예측 feature에서 제외한다.",
        "- 다음 검증은 같은 GPU의 비사건 시간대를 activity·요일·시간대로 매칭한 control window 비교다.",
    ]
    (OUT / "EVENT_WINDOW_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    events = pd.read_csv(OUT / "event_index.csv")
    features = pd.read_csv(OUT / "event_window_features.csv")
    util = features[features["metric"] == "gpu_util"].set_index("event_id")["baseline_60_30m"]
    active_ids = util[util >= 10].index
    effects = active_effects(events, features, active_ids)
    curves = active_curves(events, active_ids)
    effects.to_csv(OUT / "event_active_effect_summary.csv", index=False)
    curves.to_csv(OUT / "event_active_short_curve_summary.csv", index=False)
    plot(curves)
    write_findings(events, effects, curves, active_ids)


if __name__ == "__main__":
    main()
