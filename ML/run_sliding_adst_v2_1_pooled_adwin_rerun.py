"""Corrected v2.1 entry point with an explicit ADWIN minimum-batch gate.

The main v2.1 runner is preserved.  This wrapper exists because the first
trial exposed that the minimum-batch contract should be enforced explicitly
at the detector boundary, not only through ADWINLite's subwindow setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import run_bidirectional_adst_fusion as base
import run_sliding_adst_v2_1_pooled_adwin as impl


class CorrectedPooledBatchADST(impl.PooledBatchADST):
    @staticmethod
    def _advance_detector(
        detector: impl.v2.ADWINLite,
        previous_state: dict[str, Any] | None,
        value: float,
    ) -> tuple[bool, bool, dict[str, Any]]:
        state = previous_state or {}
        if len(detector.values) < impl.ADWIN_MIN_BATCHES - 1:
            detector.update(value)
            payload = detector.to_dict()
            payload.update(
                {
                    "trigger_streak": 0,
                    "cooldown_remaining": max(0, int(state.get("cooldown_remaining", 0))),
                }
            )
            return False, False, payload
        return impl.PooledBatchADST._advance_detector(detector, state, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrected All-XID ADST v2.1 pooled validation and batch ADWIN"
    )
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--test-stride-bins", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/2026-09-14_All-XID_ADST_v2_1_PooledValidation_BatchADWIN_Rerun_01",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = base.find_data_dir()
    cache_dir = base.PROJECT_ROOT / "outputs" / "branch1" / "cache"
    if not cache_dir.exists():
        cache_dir = base.PARENT_ROOT / "outputs" / "branch1" / "cache"
    engine = base.UnifiedDataEngine(data_dir=data_dir, cache_dir=cache_dir)
    _, history_map, gt_matrix = engine.load_all_xid_ledger()
    pipeline = CorrectedPooledBatchADST(
        engine=engine,
        history_map=history_map,
        gt_matrix=gt_matrix,
        output_dir=base.PROJECT_ROOT / args.output_dir,
        retrain_cadence_hours=args.cadence_hours,
        negative_ratio=args.negative_ratio,
        test_stride_bins=args.test_stride_bins,
        resume=args.resume,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
