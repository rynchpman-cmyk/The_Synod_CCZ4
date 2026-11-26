# The_Synod_CCZ4

The last black hole has evaporated. The Urumes is still here. Somewhere behind the Wall, the Synod advances spacetime one time step at a time.

Welcome to the Synod's test bench.

## What the Synod is solving

In the Urumes setting, the Synod **is** the ship's CCZ4 solver. The ship is its domain. Sensors tell it what state that domain is in, and actuators let it change flows, fields, and boundary conditions. Between those two jobs, it has to keep the equations and the ship in working order.

**The_Synod_CCZ4** is a real-world numerical experiment bench modeled on that fictional solver. Its purpose is to explore **Physical Nulloth**, the invented relativistic field theory the Synod evolves aboard the Urumes. It uses **CCZ4**, the conformal and covariant Z4 formulation of Einstein's equations, which evolves constraint violations and gives the solver a way to damp them.

Physical Nulloth couples ordinary matter to a second current called the Null, a scalar coherence field, and two mixed massive vector fields. Here you can change a coupling, seed a perturbation, and watch what the equations do. Some experiments isolate a single mechanism. Others evolve coupled fields or audit the characteristic speeds and eigenvectors to see where the equations support a well-posed initial-value problem. The experiments tell us how this fictional model behaves under the conditions we test.

## Your first shift

Use Python 3.10 or newer and run these commands from the repository root.

```sh
python -m pip install -e .
python experiments/run_nulloth_homogeneous_growth.py
```

That run seeds a tiny scalar perturbation in a homogeneous state and compares its measured growth rate to the prediction from the linearized equations. The JSON output gives you both rates and their relative error. Try changing `--amplitude` and see how far the linear prediction holds.

The solver CLI is available through `python -m causal_repair --help`. Run the test suite with `python -m unittest discover -s tests -v`.

## The lab notebook

- **What happens when two characteristic speeds meet?** The Hadamard pinch campaign follows the collision through longitudinal and oblique directions. Along the flow the modes meet with independent eigenvectors. At an oblique angle the eigenvectors can coalesce. [Read the campaign](experiments/hadamard_pinch/HADAMARD_PINCH_CAMPAIGN.md)
- **What else is on the bench?** The experiment notebook collects questions about growth, mode coupling, transient amplification, and whether trajectories can reach the states found in the characteristic scans. [Open the experiment notebook](Nulloth%20exp.txt)
- **Want the equations or the ship?** The specification gives the Physical Nulloth model, the solver guide describes the numerical machinery, and the lore book takes you aboard the Urumes. [Physical Nulloth specification](NULLOTH_SPEC.md) · [Solver guide](HOW_IT_WORKS.md) · [Urumes lore book](WORLDBUILDING_THE_URUMES.txt)
