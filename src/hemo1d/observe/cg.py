from __future__ import annotations

import numpy as np
from dolfinx import fem

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.cg.discretization import CGFEMDiscretization
from hemo1d.solvers.cg.state import CGState
from hemo1d.observe.history import ProbePoint, ProbeSample


def evaluate_cg_scalar_nearest_dof(
    discretization: CGFEMDiscretization,
    function: fem.Function,
    coordinate: float,
) -> float:
    """
    Evaluate a scalar CG function at the nearest DOF to a given coordinate.

    Args:
        discretization: CG spatial discretization object with cached DOF coordinates.
        function: dolfinx Function object to evaluate.
        coordinate: Spatial coordinate (z) where evaluation is requested.

    Returns:
        Function value at the nearest DOF to the given coordinate.

    Notes:
        Uses cached DOF coordinates for O(1) lookups. This is simple and robust for now.
        Future improvements could use exact point evaluation through bounding-box trees.
    """
    _, coords = discretization.dof_coordinates_sorted()
    index = int(np.argmin(np.abs(coords - coordinate)))
    # Need to map back to original DOF index
    dofs, _ = discretization.dof_coordinates_sorted()
    dof_index = int(dofs[index])
    return float(function.x.array[dof_index])


def sample_cg_state_at_probe(
    *,
    discretization: CGFEMDiscretization,
    physics: Hemo1DPhysics,
    state: CGState,
    probe: ProbePoint,
    time: float,
) -> ProbeSample:
    """
    Sample area, flow rate, and pressure from a CG state at a probe point.

    Args:
        discretization: CG spatial discretization object.
        physics: Physics model for pressure computation.
        state: Current CG state (area and flow rate fields).
        probe: ProbePoint specifying vessel ID, name, and spatial coordinate.
        time: Current simulation time.

    Returns:
        ProbeSample containing time, area, flow_rate, and pressure at the probe.

    Raises:
        ValueError: If probe coordinate is outside the vessel domain [0, length].
    """
    if probe.coordinate < 0.0 or probe.coordinate > discretization.length:
        raise ValueError(
            f"Probe {probe.name!r} on vessel {probe.vessel_id!r} has coordinate "
            f"{probe.coordinate}, outside [0, {discretization.length}]."
        )

    area = evaluate_cg_scalar_nearest_dof(
        discretization=discretization,
        function=state.A,  # type: ignore[arg-type]
        coordinate=probe.coordinate,
    )
    flow_rate = evaluate_cg_scalar_nearest_dof(
        discretization=discretization,
        function=state.Q,  # type: ignore[arg-type]
        coordinate=probe.coordinate,
    )
    pressure = float(physics.pressure(area))

    return ProbeSample(
        time=time,
        vessel_id=probe.vessel_id,
        name=probe.name,
        coordinate=probe.coordinate,
        area=area,
        flow_rate=flow_rate,
        pressure=pressure,
    )


class CGProbeRecorder:
    """
    Probe recorder for one CG vessel.

    This is still useful for single-vessel runs, but now probes include
    vessel_id, so the same ProbeHistory format works for networks too.
    """

    def __init__(
        self,
        *,
        vessel_id: str,
        discretization: CGFEMDiscretization,
        physics: Hemo1DPhysics,
        probes: list[ProbePoint],
    ) -> None:
        self.vessel_id = vessel_id
        self.discretization = discretization
        self.physics = physics
        self.probes = probes

        self._validate_probes()

    def _validate_probes(self) -> None:
        for probe in self.probes:
            if probe.vessel_id != self.vessel_id:
                raise ValueError(
                    f"Probe {probe.name!r} belongs to vessel {probe.vessel_id!r}, "
                    f"but this recorder is for vessel {self.vessel_id!r}."
                )

            if probe.coordinate < 0.0 or probe.coordinate > self.discretization.length:
                raise ValueError(
                    f"Probe {probe.name!r} has coordinate {probe.coordinate}, "
                    f"outside [0, {self.discretization.length}]."
                )

    def sample(
        self,
        state: CGState,
        time: float,
    ) -> list[ProbeSample]:
        return [
            sample_cg_state_at_probe(
                discretization=self.discretization,
                physics=self.physics,
                state=state,
                probe=probe,
                time=time,
            )
            for probe in self.probes
        ]
