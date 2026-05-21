from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class StateAtPoint:
    """
    Physical state at a single point of a vessel.

    area:
        Cross-sectional area A.

    flow_rate:
        Volumetric flow rate Q.

    This is used by boundary conditions, junctions, diagnostics, and later
    both CG and DG discretizations.
    """

    area: float
    flow_rate: float


@dataclass(frozen=True)
class EndpointData:
    """
    State and spatial derivative data at one vessel endpoint.

    The compatibility condition needs both:

        U = [A, Q]
        dU/dz = [dA/dz, dQ/dz]

    evaluated at the endpoint at time t^n.
    """

    state: StateAtPoint
    d_area_dz: float
    d_flow_rate_dz: float


@dataclass(frozen=True)
class BoundaryState:
    """
    State prescribed or computed at a vessel endpoint.

    In CG this will be imposed as an endpoint value.

    In DG this will be used as the exterior trace for the numerical flux.
    """

    area: float
    flow_rate: float


class EndpointSide(str, Enum):
    """
    Side of a vessel endpoint.

    LEFT corresponds to z = 0.
    RIGHT corresponds to z = L.
    """

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class VesselEndpoint:
    """
    Identifier for one endpoint of one vessel.

    This is deliberately independent of CG/DG. Later, the network solver will
    use this to connect boundary conditions and junctions to vessel endpoints.
    """

    vessel_id: str
    side: EndpointSide

    @property
    def outward_normal_sign(self) -> int:
        """
        Sign of the outward normal in the vessel coordinate.

        For a 1D vessel oriented from z=0 to z=L:

            left endpoint  z=0: outward normal is -1
            right endpoint z=L: outward normal is +1
        """
        if self.side == EndpointSide.LEFT:
            return -1
        return 1

    def outward_flow(self, flow_rate: float) -> float:
        """
        Flow rate leaving the vessel through this endpoint.

        If Q is positive in the increasing-z direction:

            at left endpoint:  outward flow = -Q
            at right endpoint: outward flow = +Q
        """
        return self.outward_normal_sign * flow_rate