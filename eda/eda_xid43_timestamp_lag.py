from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EVENT_DIR = ROOT / "outputs" / "eda_event_windows"
MATCH_DIR = ROOT / "outputs" / "eda_matched_controls"
OUTPUT = ROOT / "outputs" / "eda_xid43_timestamp_lag"
OFFSETS = np.arange(-60, 60)


def trailing_low_minutes(values: np.ndarray, threshold: float, lookback: int = 30) -> np.ndarray:
    """Consecutive low one-minute bins immediately before the XID time."""
    segment = values[:, 60 - lookback : 60]
    result = np.full(len(segment), np.nan, dtype=float)
    for index, row in enumerate(segment):
        if not np.isfinite(row[-1]):
            continue
        run = 0
        for value in row[::-1]:
            if not np.isfinite(value) or value > threshold:
                break
            run += 1
        result[index] = run
    return result


def bootstrap_mean_ci(values: np.ndarray, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(43)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def curve(metric: str, event_ids: np.ndarray, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event = np.load(EVENT_DIR / f"{metric}_event_curves.npz")["short"][event_ids]
    control = np.load(MATCH_DIR / f"candidate_{metric}_curves.npz")["values"][point_ids]
    return event, control


def build_pair_table(matched: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    selected = matched[(matched["event_activity"] == "high") & matched["xid_code"].isin([31, 43])].copy()
    selected["start_time"] = pd.to_datetime(selected["start_time"], utc=True)
    event_ids = selected["event_id"].to_numpy(dtype=np.int32)
    point_ids = selected["point_id"].to_numpy(dtype=np.int32)
    curves = {metric: curve(metric, event_ids, point_ids) for metric in ["gpu_util", "power_usage", "gpu_temp", "fb_used"]}
    result = selected[
        ["match_id", "event_id", "cluster_id", "gpu_id", "node_id", "xid_code", "start_time", "control_time"]
    ].reset_index(drop=True)
    for threshold in [5, 10, 20]:
        event_lag = trailing_low_minutes(curves["gpu_util"][0], threshold)
        control_lag = trailing_low_minutes(curves["gpu_util"][1], threshold)
        result[f"event_terminal_low_minutes_u{threshold}"] = event_lag
        result[f"control_terminal_low_minutes_u{threshold}"] = control_lag
    result["event_inferred_low_onset_u10"] = result["start_time"] - pd.to_timedelta(
        result["event_terminal_low_minutes_u10"], unit="m"
    )
    return result, curves


def sensitivity(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in [31, 43]:
        code_data = pair_table[pair_table["xid_code"] == code]
        for threshold in [5, 10, 20]:
            event_col = f"event_terminal_low_minutes_u{threshold}"
            control_col = f"control_terminal_low_minutes_u{threshold}"
            clusters = code_data.groupby("cluster_id")[[event_col, control_col]].median().dropna()
            for duration in [1, 3, 5, 10]:
                event_flag = clusters[event_col].ge(duration).astype(float)
                control_flag = clusters[control_col].ge(duration).astype(float)
                difference = (event_flag - control_flag).to_numpy()
                low, high = bootstrap_mean_ci(difference)
                rows.append(
                    {
                        "xid_code": code,
                        "util_threshold_pct": threshold,
                        "minimum_terminal_duration_min": duration,
                        "gpu_pairs": len(code_data),
                        "node_event_clusters": len(clusters),
                        "event_fraction": event_flag.mean(),
                        "control_fraction": control_flag.mean(),
                        "paired_risk_difference": difference.mean(),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "event_lag_median_all_min": clusters[event_col].median(),
                        "control_lag_median_all_min": clusters[control_col].median(),
                        "event_lag_q75_all_min": clusters[event_col].quantile(0.75),
                        "control_lag_q75_all_min": clusters[control_col].quantile(0.75),
                    }
                )
    return pd.DataFrame(rows)


def multimetric_confirmation(pair_table: pd.DataFrame, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for metric, (event_values, control_values) in curves.items():
        for source, values in [("event", event_values), ("control", control_values)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline = np.nanmedian(values[:, :30], axis=1)
                final3 = np.nanmedian(values[:, 57:60], axis=1)
            frame = pair_table[["cluster_id", "xid_code"]].copy()
            frame["change_baseline_to_final3"] = final3 - baseline
            frame["terminal_low_3m"] = pair_table[f"{source}_terminal_low_minutes_u10"].ge(3)
            for code in [31, 43]:
                selected = frame[(frame["xid_code"] == code) & frame["terminal_low_3m"]]
                cluster_values = selected.groupby("cluster_id")["change_baseline_to_final3"].median().dropna()
                rows.append(
                    {
                        "xid_code": code,
                        "source": source,
                        "metric": metric,
                        "terminal_low_gpu_pairs": len(selected),
                        "node_event_clusters": len(cluster_values),
                        "median_change": cluster_values.median(),
                        "q25_change": cluster_values.quantile(0.25),
                        "q75_change": cluster_values.quantile(0.75),
                    }
                )
    return pd.DataFrame(rows)


def cluster_synchrony(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = pair_table[pair_table["xid_code"] == 43].copy()
    for cluster_id, group in data.groupby("cluster_id"):
        detected = group[group["event_terminal_low_minutes_u10"].ge(1)]
        onset_spread = np.nan
        if len(detected) >= 2:
            onset_spread = (
                detected["event_inferred_low_onset_u10"].max()
                - detected["event_inferred_low_onset_u10"].min()
            ).total_seconds() / 60
        rows.append(
            {
                "cluster_id": cluster_id,
                "node_id": group["node_id"].iloc[0],
                "gpu_pairs": len(group),
                "terminal_low_3m_gpu_fraction": group["event_terminal_low_minutes_u10"].ge(3).mean(),
                "median_terminal_low_minutes_u10": group["event_terminal_low_minutes_u10"].median(),
                "detected_gpu_count": len(detected),
                "inferred_onset_spread_min": onset_spread,
            }
        )
    return pd.DataFrame(rows)


def plot_results(pair_table: pd.DataFrame, sensitivity_table: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(4)
    durations = [1, 3, 5, 10]
    width = 0.18
    for index, (code, source, color) in enumerate(
        [(43, "event", "tab:orange"), (43, "control", "tab:gray"), (31, "event", "tab:blue"), (31, "control", "silver")]
    ):
        data = sensitivity_table[
            (sensitivity_table["xid_code"] == code)
            & (sensitivity_table["util_threshold_pct"] == 10)
        ].set_index("minimum_terminal_duration_min")
        values = [data.loc[duration, f"{source}_fraction"] for duration in durations]
        axes[0].bar(positions + (index - 1.5) * width, values, width=width, label=f"XID {code} {source}", color=color)
    axes[0].set_xticks(positions, durations)
    axes[0].set_xlabel("Consecutive low-utilization minutes before timestamp")
    axes[0].set_ylabel("Fraction of node-event clusters")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Terminal GPU_UTIL <= 10%")

    for code, color in [(31, "tab:blue"), (43, "tab:orange")]:
        data = pair_table[pair_table["xid_code"] == code].groupby("cluster_id")[
            "event_terminal_low_minutes_u10"
        ].median().dropna()
        axes[1].hist(data, bins=np.arange(-0.5, 31.5, 1), alpha=0.55, density=True, label=f"XID {code}", color=color)
    axes[1].set_xlabel("Estimated inactivity before XID timestamp (min)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("High-activity events; 30-minute cap")
    axes[1].legend()
    figure.suptitle("Telemetry evidence for delayed XID timestamp")
    figure.tight_layout()
    figure.savefig(OUTPUT / "xid43_timestamp_lag.png", dpi=170)
    plt.close(figure)


def write_report(sensitivity_table: pd.DataFrame, multimetrics: pd.DataFrame, synchrony: pd.DataFrame) -> None:
    primary = sensitivity_table[
        (sensitivity_table["xid_code"] == 43)
        & (sensitivity_table["util_threshold_pct"] == 10)
        & (sensitivity_table["minimum_terminal_duration_min"] == 3)
    ].iloc[0]
    reference = sensitivity_table[
        (sensitivity_table["xid_code"] == 31)
        & (sensitivity_table["util_threshold_pct"] == 10)
        & (sensitivity_table["minimum_terminal_duration_min"] == 3)
    ].iloc[0]
    confirm = multimetrics[
        (multimetrics["xid_code"] == 43) & (multimetrics["source"] == "event")
    ][["metric", "node_event_clusters", "median_change", "q25_change", "q75_change"]]
    eligible_sync = synchrony[synchrony["detected_gpu_count"] >= 2]
    sync_fraction = eligible_sync["inferred_onset_spread_min"].le(2).mean() if len(eligible_sync) else np.nan
    lines = [
        "# XID 43 timestamp-lag validation",
        "",
        "## Operational definition",
        "",
        "- Primary population: matched pairs whose -60~-30 min GPU_UTIL baseline was high (>=50%).",
        "- Estimated lag: consecutive one-minute bins with GPU_UTIL <=10% immediately before the first XID timestamp.",
        "- Primary delayed-state criterion: at least 3 consecutive low-utilization minutes before XID.",
        "- Control: same GPU, weekday, time, and baseline activity matched non-event window.",
        "- XID 31 is included as a reference failure type.",
        "",
        "## Primary result",
        "",
        f"- XID 43: {primary.event_fraction:.1%} of {int(primary.node_event_clusters)} node-event clusters were already low for >=3 min; matched controls {primary.control_fraction:.1%}.",
        f"- Paired risk difference: {primary.paired_risk_difference:+.1%} (cluster bootstrap 95% CI {primary.bootstrap_ci_low:+.1%} to {primary.bootstrap_ci_high:+.1%}).",
        f"- XID 31 reference: event {reference.event_fraction:.1%}, control {reference.control_fraction:.1%}, risk difference {reference.paired_risk_difference:+.1%}.",
        f"- XID 43 cluster-level lag median {primary.event_lag_median_all_min:.1f} min, 75th percentile {primary.event_lag_q75_all_min:.1f} min; values are capped at 30 min.",
        "",
        "## Multimetric confirmation among terminal-low cases",
        "",
        confirm.to_markdown(index=False),
        "",
        "## Within-node synchrony",
        "",
        f"- XID 43 clusters with at least two detected GPU onsets: {len(eligible_sync)}.",
        f"- Of these, inferred low-state onsets occurred within 2 min across GPUs in {sync_fraction:.1%} of clusters." if len(eligible_sync) else "- Too few multi-GPU detected clusters for a synchrony estimate.",
        "",
        "## Interpretation boundary",
        "",
        "- A higher pre-XID terminal-low rate than matched control supports that the logged XID time can follow functional inactivity.",
        "- It does not identify the exact physical fault time. A normal job finish, reset, or delayed monitoring/export can produce the same telemetry ordering.",
        "- trace_seren.csv cannot directly resolve this because it has node counts but no allocated node_id or gpu_id.",
        "- Definitive validation needs scheduler allocation logs or NVIDIA kernel/journal timestamps joined by node and GPU.",
    ]
    (OUTPUT / "XID43_TIMESTAMP_LAG_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check() -> None:
    sample = np.array([[50, 10, 5, 0], [50, 0, 20, 0], [50, 0, 0, np.nan]], dtype=float)
    padded = np.full((3, 120), 100.0)
    padded[:, 56:60] = sample
    result = trailing_low_minutes(padded, 10, lookback=4)
    assert result[0] == 3 and result[1] == 1 and np.isnan(result[2])


def main() -> None:
    self_check()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCH_DIR / "matched_pairs.csv")
    pair_table, curves = build_pair_table(matched)
    sensitivity_table = sensitivity(pair_table)
    multimetrics = multimetric_confirmation(pair_table, curves)
    synchrony = cluster_synchrony(pair_table)
    pair_table.to_csv(OUTPUT / "xid43_lag_per_pair.csv", index=False)
    sensitivity_table.to_csv(OUTPUT / "xid43_lag_sensitivity.csv", index=False)
    multimetrics.to_csv(OUTPUT / "xid43_multimetric_confirmation.csv", index=False)
    synchrony.to_csv(OUTPUT / "xid43_cluster_synchrony.csv", index=False)
    plot_results(pair_table, sensitivity_table)
    write_report(sensitivity_table, multimetrics, synchrony)
    print(f"[done] {OUTPUT}")


if __name__ == "__main__":
    main()
