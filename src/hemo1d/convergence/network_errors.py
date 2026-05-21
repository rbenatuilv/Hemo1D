from __future__ import annotations

import numpy as np

from hemo1d.convergence.errors import l2_error_1d, observed_orders
from hemo1d.convergence.network_snapshots import (
    NetworkSnapshotHistory,
    NetworkSolutionSnapshot,
)
from hemo1d.convergence.reports import ConvergenceLevel, ConvergenceErrorRow
from hemo1d.convergence.richardson import richardson_extrapolate_snapshot


def network_l2_error_at_time(
    solution: NetworkSolutionSnapshot,
    reference: NetworkSolutionSnapshot,
    field: str,
) -> float:
    """
    Compute network L2 error at one time snapshot.

    Args:
        solution: Network snapshot (computed solution).
        reference: Reference network snapshot (e.g., Richardson extrapolate).
        field: Field name to compare ("area" or "flow_rate").

    Returns:
        Network L2 error: sqrt(sum_vessels ||u_vessel - u_ref_vessel||²_L2)
    """
    total_squared = 0.0

    for vessel_id, snapshot in solution.vessel_snapshots.items():
        ref_snapshot = reference.vessel_snapshots[vessel_id]

        values = getattr(snapshot, field)
        ref_values = getattr(ref_snapshot, field)

        err = l2_error_1d(values, ref_values, snapshot.z)
        total_squared += err * err

    return float(np.sqrt(total_squared))


def linf_time_network_l2_error(
    solution: NetworkSnapshotHistory,
    reference: NetworkSnapshotHistory,
    field: str,
    atol_time: float = 1.0e-12,
) -> float:
    """
    Compute maximum network L2 error over time.

    Args:
        solution: Network snapshot history (computed solution).
        reference: Reference network history (e.g., Richardson extrapolate).
        field: Field name to compare ("area" or "flow_rate").
        atol_time: Time tolerance for matching snapshots across histories.

    Returns:
        max_t sqrt(sum_vessels ||field_v - field_ref_v||²_L2)
    """
    errors = []

    for snapshot in solution.snapshots:
        ref_snapshot = reference.snapshot_at_time(snapshot.time, atol=atol_time)

        errors.append(
            network_l2_error_at_time(
                solution=snapshot,
                reference=ref_snapshot,
                field=field,
            )
        )

    return max(errors) if errors else 0.0


def richardson_extrapolate_network_snapshot(
    coarse: NetworkSolutionSnapshot,
    fine: NetworkSolutionSnapshot,
    expected_order: float,
    delta: float = 2.0,
    atol_time: float = 1.0e-12,
) -> NetworkSolutionSnapshot:
    """
    Richardson-extrapolate all vessel snapshots at one time point.

    Args:
        coarse: Coarse-grid network snapshot.
        fine: Fine-grid network snapshot (at the same time).
        expected_order: Expected convergence order (typically 2.0 for linear CG).
        delta: Mesh refinement ratio (typically 2.0).
        atol_time: Time tolerance for validating snapshot alignment.

    Returns:
        Extrapolated network snapshot with higher accuracy estimate.

    Raises:
        ValueError: If snapshots are at different times or have mismatched vessel sets.
    """
    if abs(coarse.time - fine.time) > atol_time:
        raise ValueError("Cannot extrapolate network snapshots at different times.")

    vessel_snapshots = {}

    for vessel_id, coarse_snapshot in coarse.vessel_snapshots.items():
        fine_snapshot = fine.vessel_snapshots[vessel_id]

        vessel_snapshots[vessel_id] = richardson_extrapolate_snapshot(
            coarse=coarse_snapshot,
            fine=fine_snapshot,
            expected_order=expected_order,
            delta=delta,
            atol_time=atol_time,
        )

    return NetworkSolutionSnapshot(
        time=coarse.time,
        vessel_snapshots=vessel_snapshots,
    )


def richardson_extrapolate_network_history(
    coarse: NetworkSnapshotHistory,
    fine: NetworkSnapshotHistory,
    expected_order: float,
    delta: float = 2.0,
    atol_time: float = 1.0e-12,
) -> NetworkSnapshotHistory:
    """
    Richardson-extrapolate a full network snapshot history.

    Args:
        coarse: Coarse-grid network history.
        fine: Fine-grid network history.
        expected_order: Expected convergence order (typically 2.0 for linear CG).
        delta: Mesh refinement ratio (typically 2.0).
        atol_time: Time tolerance for matching snapshots between histories.

    Returns:
        Extrapolated network history providing higher-accuracy estimates.

    Notes:
        Extrapolation is performed at each time point common to both histories.
    """
    extrapolated = NetworkSnapshotHistory()

    for coarse_snapshot in coarse.snapshots:
        fine_snapshot = fine.snapshot_at_time(coarse_snapshot.time, atol=atol_time)

        extrapolated.snapshots.append(
            richardson_extrapolate_network_snapshot(
                coarse=coarse_snapshot,
                fine=fine_snapshot,
                expected_order=expected_order,
                delta=delta,
                atol_time=atol_time,
            )
        )

    return extrapolated
def compute_network_richardson_error_rows(
    level_histories: dict[str, NetworkSnapshotHistory],
    levels: list[ConvergenceLevel],
    expected_order: float = 2.0,
    delta: float = 2.0,
) -> list[ConvergenceErrorRow]:
    """
    Compute Richardson-estimated network errors for consecutive mesh refinement levels.

    Args:
        level_histories: Mapping from level name to network snapshot history.
        levels: List of ConvergenceLevel objects in increasing refinement order.
        expected_order: Expected convergence order for error computation.
        delta: Mesh refinement ratio between consecutive levels.

    Returns:
        List of ConvergenceErrorRow objects, one for each coarse-fine level pair.
        Each row contains area and flow_rate errors for that refinement level.
    """
    rows: list[ConvergenceErrorRow] = []

    for coarse_level, fine_level in zip(levels[:-1], levels[1:]):
        coarse = level_histories[coarse_level.name]
        fine = level_histories[fine_level.name]

        rich = richardson_extrapolate_network_history(
            coarse=coarse,
            fine=fine,
            expected_order=expected_order,
            delta=delta,
        )

        area_error = linf_time_network_l2_error(
            solution=coarse,
            reference=rich,
            field="area",
        )
        flow_error = linf_time_network_l2_error(
            solution=coarse,
            reference=rich,
            field="flow_rate",
        )

        rows.append(
            ConvergenceErrorRow(
                level_name=coarse_level.name,
                num_cells=coarse_level.num_cells,
                dt=coarse_level.dt,
                h_like=coarse_level.h_like,
                area_error=area_error,
                flow_rate_error=flow_error,
            )
        )

    return rows


def network_observed_orders(
    rows: list[ConvergenceErrorRow],
    delta: float = 2.0,
) -> tuple[list[float], list[float]]:
    """
    Compute observed convergence orders from network error rows.

    Args:
        rows: List of ConvergenceErrorRow objects from consecutive refinement levels.
        delta: Mesh refinement ratio (typically 2.0).

    Returns:
        Tuple of (area_orders, flow_orders) lists, one order per consecutive pair.
    """
    area_errors = [row.area_error for row in rows]
    flow_errors = [row.flow_rate_error for row in rows]

    return (
        observed_orders(area_errors, delta=delta),
        observed_orders(flow_errors, delta=delta),
    )