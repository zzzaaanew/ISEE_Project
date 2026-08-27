from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import prepare_xid_observability_compact as pipeline


def finalize() -> None:
    output = pipeline.OUTPUT
    path = output / "fault_tape_v2.csv"
    fault = pd.read_csv(path)
    fault["xid_episode_duration_status"] = np.where(
        fault["end_right_censored"].fillna(True),
        "right_censored_lower_bound",
        "observed_xid_transition",
    )
    fault["replay_downtime_source"] = "scenario_parameter_required"
    fault.to_csv(path, index=False)

    report_path = output / "XID_OBSERVABILITY_FINDINGS.md"
    report = report_path.read_text(encoding="utf-8").split("\n## Downtime 해석", 1)[0].rstrip()
    right = int(fault["end_right_censored"].fillna(True).sum())
    report += (
        "\n\n## Downtime 해석\n\n"
        f"- {right:,}/{len(fault):,} episode는 종료가 right-censored되어 기록 duration이 오류 지속시간의 하한이다.\n"
        "- XID episode duration은 GPU downtime과 동일하지 않다. Blox 복구시간은 별도 시나리오 파라미터로 두고 민감도 분석한다.\n"
        "- `xid_episode_duration_status`와 `replay_downtime_source` 열로 이 제한을 명시했다.\n"
    )
    report_path.write_text(report, encoding="utf-8")

    assert fault["replay_downtime_source"].eq("scenario_parameter_required").all()
    print(f"Finalized {len(fault):,} Fault Tape rows; right-censored={right:,}")


if __name__ == "__main__":
    if "--rescan" in sys.argv:
        pipeline.main()
    finalize()
