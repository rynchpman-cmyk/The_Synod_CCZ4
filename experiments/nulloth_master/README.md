# Nulloth Observatory

This directory contains the provenance and evidence machinery behind the
Nulloth Master Spec. It separates three things that must never be conflated:

1. a registered scientific question;
2. an immutable execution record;
3. a claim admitted to the Evidence Ledger.

## Campaign lifecycle

A campaign manifest records its prediction, falsifier, controls, decisive
observables, source inputs, deterministic seed, and exact command. The
observatory hashes the manifest and every declared input to produce a stable
run identity. Repeated executions receive separate attempt directories and do
not overwrite prior results.

Run the registered homogeneous-growth campaign with:

```powershell
$env:PYTHONPATH='src'
python experiments\nulloth_master\observatory.py run `
  experiments\nulloth_master\campaigns\NUL-MOD-001.json
```

Validate manifests, the artifact index, and the Evidence Ledger with:

```powershell
python experiments\nulloth_master\observatory.py validate
```

Regenerate the compact evidence tables with:

```powershell
python experiments\nulloth_master\observatory.py render
```

Run records are written below `runs/nulloth_master`. A run record captures the
command, source fingerprint, environment, precision, dependency versions,
timestamps, duration, exit status, stdout, stderr, and hashes of captured
outputs. An unsuccessful attempt remains evidence and is never silently
deleted.

## Supersession law

Evidence is append-only. A new campaign may supersede an older claim, but it
does not replace its artifacts. Superseded claims receive `REVOKED` status and
identify their replacement. Changed source inputs produce a new run identity;
they do not mutate the identity of an earlier run.

