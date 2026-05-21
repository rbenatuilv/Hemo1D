from __future__ import annotations

import numpy as np

from hemo1d.convergence.snapshots import SnapshotHistory, SolutionSnapshot


def richardson_extrapolate_values(
    coarse_values: np.ndarray,
    fine_values: np.ndarray,
    expected_order: float,
    delta: float = 2.0,
) -> np.ndarray:
    """
    Richardson extrapolation from two levels.

    If fine_values approximates u(h/delta) and coarse_values approximates u(h),
    then:

        R = (delta^p * u(h/delta) - u(h)) / (delta^p - 1)

    where p is the expected convergence order.
    """
    factor = delta**expected_order

    return (factor * fine_values - coarse_values) / (factor - 1.0)


def richardson_extrapolate_snapshot(
    coarse: SolutionSnapshot,
    fine: SolutionSnapshot,
    expected_order: float,
    delta: float = 2.0,
    atol_time: float = 1.0e-12,
) -> SolutionSnapshot:
    """
    Richardson-extrapolate one snapshot.

    Both snapshots must already be sampled on the same z grid.
    """
    if abs(coarse.time - fine.time) > atol_time:
        raise ValueError("Cannot extrapolate snapshots at different times.")

    if not np.allclose(coarse.z, fine.z):
        raise ValueError("Cannot extrapolate snapshots on different spatial grids.")

    area = richardson_extrapolate_values(
        coarse.area,
        fine.area,
        expected_order=expected_order,
        delta=delta,
    )
    flow_rate = richardson_extrapolate_values(
        coarse.flow_rate,
        fine.flow_rate,
        expected_order=expected_order,
        delta=delta,
    )
    pressure = richardson_extrapolate_values(
        coarse.pressure,
        fine.pressure,
        expected_order=expected_order,
        delta=delta,
    )

    return SolutionSnapshot(
        time=coarse.time,
        z=coarse.z.copy(),
        area=area,
        flow_rate=flow_rate,
        pressure=pressure,
    )


def richardson_extrapolate_history(
    coarse: SnapshotHistory,
    fine: SnapshotHistory,
    expected_order: float,
    delta: float = 2.0,
    atol_time: float = 1.0e-12,
) -> SnapshotHistory:
    """
    Richardson-extrapolate a full snapshot history.
    """
    extrapolated = SnapshotHistory()

    for coarse_snapshot in coarse.snapshots:
        fine_snapshot = fine.snapshot_at_time(coarse_snapshot.time, atol=atol_time)

        extrapolated.snapshots.append(
            richardson_extrapolate_snapshot(
                coarse=coarse_snapshot,
                fine=fine_snapshot,
                expected_order=expected_order,
                delta=delta,
                atol_time=atol_time,
            )
        )

    return extrapolated