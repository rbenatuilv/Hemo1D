from __future__ import annotations

from abc import ABC, abstractmethod

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide


class BoundaryCondition(ABC):
    """
    Abstract boundary condition for one vessel endpoint.

    The boundary layer returns a BoundaryState:

        A_boundary, Q_boundary

    In CG:
        this state is imposed strongly at the endpoint.

    In DG:
        this state will later be used as an exterior trace for the numerical flux.
    """

    @abstractmethod
    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        raise NotImplementedError