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
class JunctionData:
    """Data for one two- or three-vessel junction."""

    endpoints: tuple[JunctionEndpointData, ...]

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if len(endpoints) not in (2, 3):
            raise ValueError("JunctionData must contain exactly 2 or 3 endpoints.")
        object.__setattr__(self, "endpoints", endpoints)


@dataclass(frozen=True)
class JunctionSolution:
    """Solved endpoint states at a two- or three-vessel junction."""

    endpoint_states: tuple[BoundaryState, ...]
    newton_result: NewtonResult

    def __post_init__(self) -> None:
        states = tuple(self.endpoint_states)
        if len(states) not in (2, 3):
            raise ValueError("JunctionSolution must contain exactly 2 or 3 endpoint states.")
        object.__setattr__(self, "endpoint_states", states)


__all__ = [
    "JunctionData",
    "JunctionEndpointData",
    "JunctionSolution",
]
