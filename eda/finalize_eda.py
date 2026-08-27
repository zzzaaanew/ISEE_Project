from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUT = Path(__file__).resolve().parent.parent / "outputs" / "eda_phase1"
plt.style.use("seaborn-v0_8-whitegrid")


def trace_plot():
    daily = pd.read_csv(OUT / "trace_daily.csv", parse_dates=["submit_date"])
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(daily["submit_date"], daily["submitted_jobs"], label="Submitted jobs")
    axes[0].plot(daily["submit_date"], daily["submitted_gpus"], label="Requested GPUs")
    axes[0].legend()
    axes[0].set_ylabel("Count")
    axes[1].plot(daily["submit_date"], daily["gpu_hours"], color="tab:green")
    axes[1].set_ylabel("GPU hours")
    axes[2].plot(daily["submit_date"], daily["failed_rate"] * 100, color="tab:red")
    axes[2].set_ylabel("Failed jobs (%)")
    axes[2].set_xlabel("Submit date (UTC)")
    figure.suptitle("Trace workload by day")
    figure.tight_layout()
    figure.savefig(OUT / "trace_workload.png", dpi=160)
    plt.close(figure)


def horizon_plot():
    balance = pd.read_csv(OUT / "risk_horizon_balance.csv")
    rates = balance["positive_rate"] * 100
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(balance["horizon"], rates)
    axis.set_yscale("log")
    axis.set_ylabel("Positive GPU-time rows (%) — log scale")
    axis.set_title("Risk-label prevalence by horizon")
    for bar, rate in zip(bars, rates):
        axis.text(bar.get_x() + bar.get_width() / 2, rate * 1.15, f"{rate:.4f}%", ha="center")
    figure.tight_layout()
    figure.savefig(OUT / "risk_horizon_balance.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    trace_plot()
    horizon_plot()
