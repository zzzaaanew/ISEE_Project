"""Entry point for the fixed Sliding Training Pareto experiment."""

from __future__ import annotations

import sys

import run_pareto_momentum_enhanced_fusion as runner


if __name__ == "__main__":
    if "--policy" not in sys.argv:
        sys.argv[1:1] = ["--policy", "sliding"]
    runner.main()

