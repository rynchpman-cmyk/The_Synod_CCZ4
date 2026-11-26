"""Run with: python examples/proca_pulse.py"""

from causal_repair.cli import main

raise SystemExit(main(["proca", "--final-time", "4", "--report-every", "50"]))

