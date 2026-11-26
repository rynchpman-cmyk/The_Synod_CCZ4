"""Run with: python examples/relaxation.py"""

from causal_repair.cli import main

raise SystemExit(main(["relax", "--deterministic", "--final-time", "5"]))

