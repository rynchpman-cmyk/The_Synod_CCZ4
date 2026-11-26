"""Validate the ordered continuum-to-production trace for Nulloth."""

from __future__ import annotations

import json
from pathlib import Path


def validate_production_trace(trace_path: Path, root: Path) -> list[str]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if trace.get("state_groups") != 27:
        errors.append("production trace must declare 27 state groups")
    seen: set[str] = set()
    allowed_kinds = {"PHYSICAL", "NUMERICAL", "IRREVERSIBLE", "VALIDATION"}
    for entry in trace.get("entries", []):
        label = entry.get("id", "unnamed production operation")
        if label in seen:
            errors.append(f"{label}: duplicate production id")
        seen.add(label)
        if entry.get("kind") not in allowed_kinds:
            errors.append(f"{label}: unknown production kind")
        for witness in [entry.get("implementation", {})] + entry.get(
            "verification", []
        ):
            path = root / witness.get("path", "")
            symbol = witness.get("symbol", "")
            if not path.is_file():
                errors.append(f"{label}: missing path {witness.get('path')}")
            elif symbol not in path.read_text(encoding="utf-8"):
                errors.append(
                    f"{label}: missing symbol {symbol!r} in {witness.get('path')}"
                )

    source = (root / "src/causal_repair/production_nulloth.py").read_text(
        encoding="utf-8"
    )
    step_start = source.index("    def step(")
    step_end = source.index("    def recommended_dt(", step_start)
    step_source = source[step_start:step_end]
    ordered_step_symbols = (
        "k1 = tuple(self.rhs",
        "result.geometry = self.ccz4.project_algebraic",
        "result.scalar.gradient = self.grid.gradient",
        "result = self._project_to_master_manifold",
        "result = self._apply_carrier_drag",
        "result = self._apply_damping",
        "self.recover(result)",
    )
    positions = [step_source.find(symbol) for symbol in ordered_step_symbols]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("production step operations do not match the declared order")

    rhs_start = source.index("    def rhs(")
    rhs_end = source.index("    def _apply_carrier_drag(", rhs_start)
    rhs_source = source[rhs_start:rhs_end]
    ordered_rhs_symbols = (
        "recovery = self.recover(state)",
        "geometry_rhs = self.ccz4.rhs",
        "matter_rhs = list(self._conservative_rhs",
        "target_charge_rhs, target_momentum_rhs = self._target_rhs",
        "field_rhs = self.system.rhs_fields",
        "scalar_rhs = self.scalar_system.rhs",
    )
    positions = [rhs_source.find(symbol) for symbol in ordered_rhs_symbols]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("production RHS operations do not match the declared order")
    return errors

