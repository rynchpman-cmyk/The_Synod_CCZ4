"""Run with: python examples/coupled_adm_proca.py"""

from causal_repair.cli import main

raise SystemExit(main(["coupled", "--final-time", "0.05", "--report-every", "5"]))

