# The Complete-Step Entropy Account of Nulloth

## Synodic identity

Campaign `NUL-NLN-005` activates a sinusoidal carrier counterflow and closes
the entropy account at complete-step boundaries:

```text
net entropy = implicit collision-map entropy + finite-volume/projection entropy
```

The collision term is the actual entropy increase written by the post-RK
implicit drag map.  The remainder is defined by the complete-step identity;
it includes entropy produced by the finite-volume conservative update and its
final constitutive projection.  It is nonnegative and must decrease with
spatial resolution.

## Proscribed accounting

`last_projection_entropy_change` accumulates corrections at provisional RK
stages as well as the retained final state.  Provisional stages are discarded,
so this raw sum is not an additive physical channel.  The campaign records its
ratio to the complete-step numerical remainder precisely to demonstrate the
overcount.  It must never be added to net or drag entropy in a Synodic budget.

## Scope

The time-step ladder tests temporal stability of the net and physical drag
terms.  The 8, 16, and 32 point ladder tests convergence of numerical entropy.
This closes the entropy account for the declared periodic counterflow trial.
It does not close gravitational energy, global energy, finite-boundary entropy
flux, or strong-field thermodynamics.
