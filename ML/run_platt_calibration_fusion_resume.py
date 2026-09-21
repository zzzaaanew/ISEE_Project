"""Resume entry point for the Platt calibration experiment after a namespace repair."""

from branch1_enhanced_features import EnhancedTelemetryEngine
import run_fusion_diversified_b1_history_b2 as integrated
import run_report_faithful_platt_calibration_fusion as runner
from run_pareto_momentum_enhanced_fusion import _pareto_counts


integrated.EnhancedTelemetryEngine = EnhancedTelemetryEngine
integrated._pareto_counts = _pareto_counts


if __name__ == "__main__":
    runner.main()
