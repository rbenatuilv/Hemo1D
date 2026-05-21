from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide, StateAtPoint
from hemo1d.solvers.time import TimeConfig


class BaseSolver(Protocol):
    """Common interface for high-level model solvers."""

    def run(
        self,
        config: TimeConfig,
        record_every: int = 1,
        probes: list[Any] | None = None,
        snapshot_sample_points_by_vessel: dict[str, np.ndarray] | None = None,
        show_progress: bool = True,
        progress_description: str = "Solving network",
    ) -> Any:
        ...


class VesselDiscretization(Protocol):
    """
    Common interface expected from a vessel spatial discretization.

    CG and DG discretizations should both implement this protocol.
    """

    @property
    def length(self) -> float:
        """
        Return vessel length.
        """
        ...

    def num_dofs(self) -> int:
        """
        Return the number of scalar degrees of freedom per variable.
        """
        ...

    def create_state(self, name: str = "") -> Any:
        """
        Create an empty state container compatible with this discretization.
        """
        ...

    def interpolate_rest_state(
        self,
        state: Any,
        physics: Hemo1DPhysics,
    ) -> None:
        """
        Set a state to rest:

            A = A0
            Q = 0
        """
        ...

    def endpoint_state(
        self,
        state: Any,
        side: EndpointSide,
    ) -> StateAtPoint:
        """
        Return A and Q at one endpoint.
        """
        ...

    def endpoint_data(
        self,
        state: Any,
        side: EndpointSide,
    ) -> EndpointData:
        """
        Return endpoint state and spatial derivatives needed by compatibility
        boundary/junction equations.
        """
        ...


class VesselStepper(Protocol):
    """
    Common interface expected from a vessel time stepper.
    """

    def compute_stable_dt(
        self,
        state: Any,
        cfl: float,
    ) -> float:
        """
        Compute a stable time step estimate for the current vessel state.
        """
        ...

    def step(
        self,
        state_n: Any,
        state_np1: Any,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        """
        Advance one vessel state from t^n to t^{n+1}.
        """
        ...


VesselStateSampler = Callable[
    [
        Any,  # concrete discretization object
        Any,  # state object
        Hemo1DPhysics,
        float,  # coordinate
    ],
    StateAtPoint,
]


VesselStateArrayExtractor = Callable[
    [
        Any,  # concrete discretization object
        Any,  # state object
    ],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


VesselStateArraySampler = Callable[
    [
        Any,  # concrete discretization object
        Any,  # state object
        Hemo1DPhysics,
        np.ndarray,  # coordinates
    ],
    tuple[np.ndarray, np.ndarray],
]
