# Active-repair showcase runs

These five trajectories were selected from a 54-case scan and rerun at twice
the scan resolution (`N=160`, `L=24`, `t=8`). All remain finite and satisfy the
algebraically reconstructed Gauss constraint to roughly `1e-18` or better.

| Name | `m_A` | `tau_star` | coupling | peak density | Signature |
|---|---:|---:|---:|---:|---|
| `radiative_control` | 0.25 | 0.15 | 0 | 0.002 | Clean quench; 99.99% of repair energy reaches `abs(x)>3.5` |
| `fast_quench_trap` | 0.25 | 0.15 | 30 | 0.08 | Fast mismatch formation; interaction energy reaches 0.695 of bare Proca energy |
| `ringing_trap` | 0.6 | 0.15 | 30 | 0.08 | Five phase reversals with a slowly dispersing localized component |
| `slow_overshoot` | 1.2 | 2.4 | 0 | 0.002 | Mismatch overshoots to 1.330 times the initial field norm |
| `interaction_dominant` | 1.2 | 2.4 | 30 | 0.08 | Interaction energy peaks at 1.183 times bare Proca energy near `t=1.45` |

The `interaction_dominant` mismatch peaks near `t=1.5`, essentially coincident
with the interaction-energy maximum, and its initial-profile projection changes
sign near `t = 1.07, 2.13, 3.98, 5.25, 6.80`.

Reproduce all trajectories:

```powershell
$env:PYTHONPATH="$PWD\src"
python experiments\repair_showcases.py all
python experiments\render_repair_showcases.py
```

Each directory contains a compressed trajectory with time-series observables
and the fields `a_z`, `astar_z`, `pi_z`, `phi`, and total repair-sector energy.
These arrays are intended to be the first data source for the visualization
dashboard.
