from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolutionSnapshot:
    """
    Spatial sample of a solution at one time.

    This data structure is discretization-agnostic. It stores values sampled on
    a common coordinate grid, independently of how the numerical solution is
    represented internally.
    """

    time: float
    z: np.ndarray
    area: np.ndarray
    flow_rate: np.ndarray
    pressure: np.ndarray


@dataclass
class SnapshotHistory:
    """
    Time history of spatial solution snapshots.
    """

    snapshots: list[SolutionSnapshot] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        return [snapshot.time for snapshot in self.snapshots]

    def snapshot_at_time(
        self,
        time: float,
        atol: float = 1.0e-12,
    ) -> SolutionSnapshot:
        for snapshot in self.snapshots:
            if abs(snapshot.time - time) <= atol:
                return snapshot

        raise KeyError(f"No snapshot found at time {time}.")


class VesselSnapshotRecorder:
    """
    Records snapshots from one generic Vessel.

    This is the preferred recorder for convergence studies because it samples
    through:

        vessel.sample_state(coordinate)

    Therefore it is independent of the discretization method. It works for CG
    now and should work for DG later, once DG vessels implement the same Vessel
    interface.
    """

    def __init__(
        self,
        *,
        vessel: Any,
        sample_points: np.ndarray,
    ) -> None:
        self.vessel = vessel
        self.sample_points = np.asarray(sample_points, dtype=float)

        if self.sample_points.ndim != 1:
            raise ValueError("sample_points must be a one-dimensional array.")

        if np.any(self.sample_points < 0.0) or np.any(self.sample_points > vessel.length):
            raise ValueError(
                f"sample_points must lie inside vessel {vessel.vessel_id!r} domain "
                f"[0, {vessel.length}]."
            )

    def sample(self, time: float) -> SolutionSnapshot:
        areas = np.empty_like(self.sample_points, dtype=float)
        flows = np.empty_like(self.sample_points, dtype=float)

        for i, coordinate in enumerate(self.sample_points):
            point_state = self.vessel.sample_state(float(coordinate))
            areas[i] = point_state.area
            flows[i] = point_state.flow_rate

        pressures = self.vessel.physics.pressure(areas)

        return SolutionSnapshot(
            time=float(time),
            z=self.sample_points.copy(),
            area=areas,
            flow_rate=flows,
            pressure=pressures,
        )