"""Run with: python examples/production.py"""

from causal_repair.cli import main

raise SystemExit(
    main(["production", "--points", "16", "--final-time", "0.01", "--report-every", "1"])
)

