from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hemo1d.convergence.snapshots import SolutionSnapshot


@dataclass(frozen=True)
class NetworkSolutionSnapshot:
    """
    Spatial snapshots of all vessels at one time.

    vessel_snapshots:
        Maps vessel_id -> SolutionSnapshot.
    """

    time: float
    vessel_snapshots: dict[str, SolutionSnapshot]


@dataclass
class NetworkSnapshotHistory:
    """
    Time history of network spatial snapshots.
    """

    snapshots: list[NetworkSolutionSnapshot] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        return [snapshot.time for snapshot in self.snapshots]

    def snapshot_at_time(
        self,
        time: float,
        atol: float = 1.0e-12,
    ) -> NetworkSolutionSnapshot:
        for snapshot in self.snapshots:
            if abs(snapshot.time - time) <= atol:
                return snapshot

        raise KeyError(f"No network snapshot found at time {time}.")


class NetworkSnapshotRecorder:
    """
    Records spatial snapshots for every vessel in a network.

    This recorder is discretization-agnostic. It expects a dictionary:

        vessel_id -> vessel object

    where each vessel has:

        vessel.sample_state(coordinate)
        vessel.physics.pressure(area)

    For every vessel, we sample A, Q and pressure on the provided spatial grid.
    """

    def __init__(
        self,
        *,
        vessels: dict[str, Any],
        sample_points_by_vessel: dict[str, np.ndarray],
    ) -> None:
        self.vessels = vessels
        self.sample_points_by_vessel = {
            vessel_id: np.asarray(points, dtype=float)
            for vessel_id, points in sample_points_by_vessel.items()
        }

        self._validate_sample_points()

    def _validate_sample_points(self) -> None:
        missing = set(self.vessels) - set(self.sample_points_by_vessel)
        extra = set(self.sample_points_by_vessel) - set(self.vessels)

        if missing:
            raise ValueError(f"Missing sample points for vessels: {sorted(missing)}.")

        if extra:
            raise ValueError(
                f"Sample points provided for unknown vessels: {sorted(extra)}."
            )

        for vessel_id, points in self.sample_points_by_vessel.items():
            vessel = self.vessels[vessel_id]

            if points.ndim != 1:
                raise ValueError(
                    f"Sample points for vessel {vessel_id!r} must be one-dimensional."
                )

            if np.any(points < 0.0) or np.any(points > vessel.length):
                raise ValueError(
                    f"Sample points for vessel {vessel_id!r} must lie inside "
                    f"[0, {vessel.length}]."
                )

    def sample(self, time: float) -> NetworkSolutionSnapshot:
        vessel_snapshots: dict[str, SolutionSnapshot] = {}

        for vessel_id, vessel in self.vessels.items():
            points = self.sample_points_by_vessel[vessel_id]

            areas, flows = vessel.sample_state_array(points)

            pressures = vessel.physics.pressure(areas)

            vessel_snapshots[vessel_id] = SolutionSnapshot(
                time=float(time),
                z=points.copy(),
                area=areas,
                flow_rate=flows,
                pressure=pressures,
            )

        return NetworkSolutionSnapshot(
            time=float(time),
            vessel_snapshots=vessel_snapshots,
        )
