from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint
from hemo1d.solvers.base import (
    VesselDiscretization,
    VesselStateArraySampler,
    VesselStateArrayExtractor,
    VesselStateSampler,
    VesselStepper,
)


@dataclass
class Vessel:
    """
    Generic vessel wrapper.

    This class is independent of CG, DG, or any other discretization method.

    It groups:
        - vessel id,
        - physics,
        - discretization object,
        - stepper object,
        - current state,
        - next state,
        - state sampling function,
        - state array extraction function.

    Discretization-specific factories should live inside their own packages,
    for example:

        hemo1d.solvers.cg.factory.create_cg_vessel
        hemo1d.solvers.dg.factory.create_dg_vessel
    """

    vessel_id: str
    physics: Hemo1DPhysics
    discretization: VesselDiscretization
    stepper: VesselStepper
    state_n: Any
    state_np1: Any
    state_sampler: VesselStateSampler
    state_array_extractor: VesselStateArrayExtractor
    state_array_sampler: VesselStateArraySampler | None = None

    def interpolate_rest_state(self) -> None:
        """
        Set the current state to rest:

            A = A0
            Q = 0
        """
        self.discretization.interpolate_rest_state(self.state_n, self.physics)

    def endpoint_data(self, side: EndpointSide) -> EndpointData:
        """
        Return state and derivative data at one vessel endpoint.
        """
        return self.discretization.endpoint_data(self.state_n, side)

    def endpoint_state(self, side: EndpointSide) -> StateAtPoint:
        """
        Return A and Q at one vessel endpoint.
        """
        return self.discretization.endpoint_state(self.state_n, side)

    def sample_state(self, coordinate: float) -> StateAtPoint:
        """
        Sample the current state at a local vessel coordinate.

        This is the key observation interface used by probes and convergence
        tools. The implementation is supplied by the discretization-specific
        vessel factory.
        """
        if coordinate < 0.0 or coordinate > self.length:
            raise ValueError(
                f"Coordinate {coordinate} is outside vessel {self.vessel_id!r} "
                f"domain [0, {self.length}]."
            )

        return self.state_sampler(
            self.discretization,
            self.state_n,
            self.physics,
            coordinate,
        )

    def state_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return arrays representing the current state.

        Returns:
            z, A, Q

        This is mainly used by diagnostics and export utilities.
        """
        return self.state_array_extractor(
            self.discretization,
            self.state_n,
        )

    def sample_state_array(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample the current state at many local vessel coordinates.

        Discretization-specific factories may provide a vectorized sampler. If
        they do not, fall back to the scalar observation interface.
        """
        if self.state_array_sampler is not None:
            return self.state_array_sampler(
                self.discretization,
                self.state_n,
                self.physics,
                coordinates,
        )

        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 1:
            raise ValueError("coordinates must be one-dimensional.")
        areas = np.empty_like(points, dtype=float)
        flows = np.empty_like(points, dtype=float)
        for i, coordinate in enumerate(points):
            point_state = self.sample_state(float(coordinate))
            areas[i] = point_state.area
            flows[i] = point_state.flow_rate
        return areas, flows

    def compute_stable_dt(self, cfl: float) -> float:
        """
        Compute this vessel's stable time step estimate.
        """
        return self.stepper.compute_stable_dt(self.state_n, cfl=cfl)

    def swap_states(self) -> None:
        """
        Swap current and next state after a successful time step.
        """
        self.state_n, self.state_np1 = self.state_np1, self.state_n

    @property
    def length(self) -> float:
        return self.discretization.length

    @property
    def num_dofs(self) -> int:
        return self.discretization.num_dofs()
