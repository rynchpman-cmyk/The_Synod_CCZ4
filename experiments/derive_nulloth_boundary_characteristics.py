"""Derive state-local incoming projectors and audit boundary coverage."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)

try:
    from experiments.audit_nulloth_ccz4_composition import (
        _background,
        _ccz4_symbol,
        _layout,
    )
except ModuleNotFoundError:
    from audit_nulloth_ccz4_composition import _background, _ccz4_symbol, _layout


@dataclass(frozen=True)
class FaceSignature:
    incoming: int
    glancing: int
    outgoing: int
    occurrences: int


@dataclass(frozen=True)
class BoundaryCharacteristicInventory:
    composed_principal_variables: int
    ccz4_incoming_modes: int
    ccz4_glancing_modes: int
    ccz4_outgoing_modes: int
    generic_physical_incoming_modes: int
    generic_physical_glancing_modes: int
    generic_physical_outgoing_modes: int
    generic_composed_incoming_modes: int
    generic_composed_glancing_modes: int
    generic_composed_outgoing_modes: int
    sampled_face_states: int
    distinct_face_signatures: int
    face_signatures: tuple[FaceSignature, ...]
    minimum_incoming_modes: int
    maximum_incoming_modes: int
    maximum_glancing_modes: int
    count_conservation_failures: int
    maximum_physical_projector_condition: float
    maximum_physical_eigenspace_residual: float
    maximum_physical_projector_idempotence_error: float
    maximum_physical_projector_commutator_error: float
    maximum_physical_projector_completeness_error: float
    ccz4_projector_condition: float
    ccz4_projector_idempotence_error: float
    ccz4_projector_commutator_error: float
    ccz4_projector_completeness_error: float
    exact_characteristic_incoming_modes_implemented: int
    componentwise_radiative_incoming_modes: int
    flux_only_material_incoming_modes: int
    maximum_incoming_modes_without_exact_data: int
    complete_coupled_boundary_qualified: int


def _clustered_projectors(
    symbol: np.ndarray, tolerance: float = 2.0e-7
) -> tuple[tuple[int, int, int], float, float, float, float, float]:
    eigenvalues = np.sort_complex(np.linalg.eigvals(symbol))
    clusters: list[list[complex]] = []
    for value in eigenvalues:
        if not clusters:
            clusters.append([complex(value)])
            continue
        center = sum(clusters[-1]) / len(clusters[-1])
        scale = max(1.0, abs(center), abs(value))
        if abs(value - center) <= tolerance * scale:
            clusters[-1].append(complex(value))
        else:
            clusters.append([complex(value)])
    identity = np.eye(symbol.shape[0], dtype=complex)
    pieces = []
    speeds = []
    maximum_residual = 0.0
    for cluster in clusters:
        center = sum(cluster) / len(cluster)
        multiplicity = len(cluster)
        _, _, right = np.linalg.svd(symbol.astype(complex) - center * identity)
        basis = right[-multiplicity:].conj().T
        pieces.append(basis)
        speeds.extend([center.real] * multiplicity)
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm((symbol - center * identity) @ basis, ord=2)),
        )
    basis = np.concatenate(pieces, axis=1)
    inverse = np.linalg.inv(basis)
    speeds_array = np.asarray(speeds)
    masks = (
        speeds_array < -1.0e-7,
        np.abs(speeds_array) <= 1.0e-7,
        speeds_array > 1.0e-7,
    )
    projectors = tuple(basis[:, mask] @ inverse[mask] for mask in masks)
    return (
        tuple(int(np.count_nonzero(mask)) for mask in masks),
        float(np.linalg.cond(basis)),
        maximum_residual,
        max(
            float(np.linalg.norm(projector @ projector - projector, ord=2))
            for projector in projectors
        ),
        max(
            float(
                np.linalg.norm(
                    symbol @ projector - projector @ symbol, ord=2
                )
            )
            for projector in projectors
        ),
        float(np.linalg.norm(sum(projectors) - identity, ord=2)),
    )


def _sign_invariant_projectors(
    symbol: np.ndarray,
) -> tuple[tuple[int, int, int], float, float, float, float, float]:
    """Project whole incoming/glancing/outgoing invariant subspaces.

    CCZ4 has large repeated-speed families.  Individual numerical eigenvectors
    inside a repeated family are ill-conditioned even though each sign-separated
    invariant subspace is regular.  QR within each sign group removes that
    arbitrary internal basis before constructing the three projectors.
    """
    eigenvalues, eigenvectors = np.linalg.eig(symbol)
    masks = (
        eigenvalues.real < -1.0e-7,
        np.abs(eigenvalues.real) <= 1.0e-7,
        eigenvalues.real > 1.0e-7,
    )
    pieces = tuple(np.linalg.qr(eigenvectors[:, mask])[0] for mask in masks)
    basis = np.concatenate(pieces, axis=1)
    inverse = np.linalg.inv(basis)
    counts = tuple(piece.shape[1] for piece in pieces)
    offsets = np.cumsum((0,) + counts)
    projectors = tuple(
        basis[:, offsets[index] : offsets[index + 1]]
        @ inverse[offsets[index] : offsets[index + 1]]
        for index in range(3)
    )
    identity = np.eye(symbol.shape[0], dtype=complex)
    maximum_residual = max(
        float(
            np.linalg.norm(
                symbol @ eigenvectors[:, mask]
                - eigenvectors[:, mask] * eigenvalues[mask][None, :],
                ord=2,
            )
        )
        for mask in masks
    )
    return (
        counts,
        float(np.linalg.cond(basis)),
        maximum_residual,
        max(
            float(np.linalg.norm(projector @ projector - projector, ord=2))
            for projector in projectors
        ),
        max(
            float(
                np.linalg.norm(
                    symbol @ projector - projector @ symbol, ord=2
                )
            )
            for projector in projectors
        ),
        float(np.linalg.norm(sum(projectors) - identity, ord=2)),
    )


def run_inventory(*, grid_points: int = 16) -> BoundaryCharacteristicInventory:
    grid = PeriodicGrid((grid_points,), (2.0 * np.pi,), method="spectral")
    coordinate = grid.coordinates()[0]
    solver, values = _background(grid)
    ccz4_generator = -1.0j * _ccz4_symbol(
        solver, values, _layout(values), coordinate, 1, 1.0e-7
    )
    ccz4 = _sign_invariant_projectors(ccz4_generator)
    ccz4_counts = ccz4[0]

    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    generic = master.coupled_characteristic_audit(
        0.65,
        0.2,
        np.array([0.04, 0.01, 0.0]),
        0.18,
        np.array([-0.03, 0.02, 0.01]),
        0.05,
        np.array([1.0, 0.0, 0.0]),
    )
    generic_physical = _clustered_projectors(
        np.linalg.solve(generic.time_matrix, generic.space_matrix)
    )[0]
    generic_composed = tuple(
        left + right for left, right in zip(ccz4_counts, generic_physical)
    )

    lower = np.array([0.4, 0.1, 0.12, -0.3, -0.16])
    upper = np.array([0.9, 0.35, 0.24, 0.3, 0.16])
    directions = tuple(
        sign * np.eye(3)[axis]
        for axis in range(3)
        for sign in (-1.0, 1.0)
    )
    signatures: Counter[tuple[int, int, int]] = Counter()
    maximum_condition = 0.0
    maximum_residual = 0.0
    maximum_idempotence = 0.0
    maximum_commutator = 0.0
    maximum_completeness = 0.0
    failures = 0
    for choices in itertools.product((0, 1), repeat=5):
        density, internal, carrier, counterflow, varphi = np.where(
            choices, upper, lower
        )
        for direction in directions:
            audit = master.coupled_characteristic_audit(
                density,
                internal,
                np.zeros(3),
                carrier,
                np.array([counterflow, 0.0, 0.0]),
                varphi,
                direction,
            )
            physical = _clustered_projectors(
                np.linalg.solve(audit.time_matrix, audit.space_matrix)
            )
            composed = tuple(
                left + right for left, right in zip(ccz4_counts, physical[0])
            )
            signatures[composed] += 1
            failures += int(sum(composed) != 48)
            maximum_condition = max(maximum_condition, physical[1])
            maximum_residual = max(maximum_residual, physical[2])
            maximum_idempotence = max(maximum_idempotence, physical[3])
            maximum_commutator = max(maximum_commutator, physical[4])
            maximum_completeness = max(maximum_completeness, physical[5])
    face_signatures = tuple(
        FaceSignature(*signature, occurrences)
        for signature, occurrences in sorted(signatures.items())
    )
    # Exact state-local characteristic projections currently cover the scalar
    # incoming field and one Z4 constraint field. Remaining CCZ4 and Proca
    # fields use componentwise radiative data; material fields use outflow
    # finite-volume fluxes rather than the derived coupled projector.
    exact = 2
    radiative = 15
    flux_only = 4
    maximum_incoming = max(item.incoming for item in face_signatures)
    return BoundaryCharacteristicInventory(
        48,
        *ccz4_counts,
        *generic_physical,
        *generic_composed,
        sum(signatures.values()),
        len(face_signatures),
        face_signatures,
        min(item.incoming for item in face_signatures),
        maximum_incoming,
        max(item.glancing for item in face_signatures),
        failures,
        maximum_condition,
        maximum_residual,
        maximum_idempotence,
        maximum_commutator,
        maximum_completeness,
        ccz4[1],
        ccz4[3],
        ccz4[4],
        ccz4[5],
        exact,
        radiative,
        flux_only,
        maximum_incoming - exact,
        0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=16)
    arguments = parser.parse_args()
    print(json.dumps(asdict(run_inventory(grid_points=arguments.grid_points)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
