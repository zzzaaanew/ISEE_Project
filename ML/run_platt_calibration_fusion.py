"""Executable entry point for the report-faithful Platt calibration runner.

The implementation lives in ``run_report_faithful_platt_calibration_fusion``.
This thin entry point exposes the EnhancedTelemetryEngine name expected by the
shared GitHub-base integration module before forwarding the CLI unchanged.
"""

from branch1_enhanced_features import EnhancedTelemetryEngine
import run_fusion_diversified_b1_history_b2 as integrated
import run_report_faithful_platt_calibration_fusion as runner


integrated.EnhancedTelemetryEngine = EnhancedTelemetryEngine


if __name__ == "__main__":
    runner.main()
