from __future__ import annotations

from dataclasses import dataclass

from hemo1d.core.newton import NewtonResult
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide


@dataclass(frozen=True)
class JunctionEndpointData:
    """
    Data associated with one vessel endpoint connected to a junction.

    physics:
        Physical model of the corresponding vessel.

    endpoint_data:
        State and spatial derivative at the endpoint, evaluated at t^n.

    side:
        LEFT or RIGHT endpoint of the vessel.

    name:
        Human-readable label, useful for debugging.
    """

    physics: Hemo1DPhysics
    endpoint_data: EndpointData
    side: EndpointSide
    name: str
    angle: float | None = None


@dataclass(frozen=True)
class BifurcationJunctionData:
    """
    Data for a simple 1-to-2 bifurcation.

    Convention:
        parent.RIGHT -> junction
        daughter1.LEFT -> junction
        daughter2.LEFT -> junction

    The implementation itself uses the endpoint side, so the orientation is not
    hard-coded in the compatibility helper.
    """

    parent: JunctionEndpointData
    daughter1: JunctionEndpointData
    daughter2: JunctionEndpointData


@dataclass(frozen=True)
class BifurcationSolution:
    """Solved endpoint states at a bifurcation."""

    parent: BoundaryState
    daughter1: BoundaryState
    daughter2: BoundaryState
    newton_result: NewtonResult


__all__ = [
    "BifurcationJunctionData",
    "BifurcationSolution",
    "JunctionEndpointData",
]
