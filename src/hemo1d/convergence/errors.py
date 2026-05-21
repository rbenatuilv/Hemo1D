from __future__ import annotations

import numpy as np

from hemo1d.convergence.snapshots import SnapshotHistory


def l2_error_1d(
    numerical: np.ndarray,
    reference: np.ndarray,
    z: np.ndarray,
) -> float:
    """
    Approximate L2(0,L) error using trapezoidal integration.

        ||u - u_ref||_L2 = sqrt(int (u-u_ref)^2 dz)
    """
    diff = numerical - reference
    return float(np.sqrt(np.trapezoid(diff * diff, z)))


def linf_time_l2_space_error(
    solution: SnapshotHistory,
    reference: SnapshotHistory,
    field: str,
    atol_time: float = 1.0e-12,
) -> float:
    """
    Compute:

        max_t || field_h(t) - field_ref(t) ||_L2(0,L)

    using sampled snapshots.

    The histories must use the same spatial sampling grid and matching times.
    """
    errors: list[float] = []

    for snapshot in solution.snapshots:
        ref_snapshot = reference.snapshot_at_time(snapshot.time, atol=atol_time)

        values = getattr(snapshot, field)
        ref_values = getattr(ref_snapshot, field)

        errors.append(l2_error_1d(values, ref_values, snapshot.z))

    return max(errors) if errors else 0.0


def observed_orders(
    errors: list[float],
    delta: float = 2.0,
) -> list[float]:
    """
    Estimate observed orders from consecutive errors:

        p_i = log(E_i / E_{i+1}) / log(delta)

    where E_i corresponds to the coarser level.
    """
    orders: list[float] = []

    for e_coarse, e_fine in zip(errors[:-1], errors[1:]):
        if e_coarse <= 0.0 or e_fine <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(float(np.log(e_coarse / e_fine) / np.log(delta)))

    return orders


def relative_error(error: float, reference_norm: float) -> float:
    """
    Compute relative error, with fallback if reference norm is zero.
    """
    if reference_norm <= 0.0:
        return error

    return error / reference_norm


def l2_norm_history(
    history: SnapshotHistory,
    field: str,
) -> float:
    """
    Compute max_t ||field(t)||_L2.
    """
    norms = []

    for snapshot in history.snapshots:
        values = getattr(snapshot, field)
        norms.append(float(np.sqrt(np.trapezoid(values * values, snapshot.z))))

    return max(norms) if norms else 0.0