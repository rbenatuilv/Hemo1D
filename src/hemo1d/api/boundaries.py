from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hemo1d.boundary import (
    BoundaryCondition,
    NonReflectingBoundary,
    PrescribedAreaBoundary,
    PrescribedFlowBoundary,
    PrescribedPressureBoundary,
)
from hemo1d.config import NetworkConfig
from hemo1d.topology.endpoint import NetworkEndpoint


ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class BoundaryAssignment:
    """Public boundary assignment stored before a solver network is built."""

    endpoint: NetworkEndpoint
    kind: str
    function: ScalarFunction | None = None


def normalize_boundary_kind(kind: str) -> str:
    return str(kind).strip().lower().replace("-", "_").replace(" ", "_")


def role_labels(role: str) -> set[str]:
    if role == "inlet":
        return {"inflow", "inlet", "velocity", "flow", "flow_rate"}
    if role == "outlet":
        return {"outflow", "outlet", "nonreflecting", "non_reflecting"}
    return set()


def make_boundary_condition(
    config: NetworkConfig,
    assignment: BoundaryAssignment,
) -> BoundaryCondition:
    kind = normalize_boundary_kind(assignment.kind)
    vessel = config.vessel(assignment.endpoint.vessel_id)

    if kind == "nonreflecting":
        return NonReflectingBoundary()

    if assignment.function is None:
        raise ValueError(f"Boundary {assignment.endpoint.label()} requires a function.")

    if kind in {"flow", "flowrate", "flow_rate", "q"}:
        return PrescribedFlowBoundary(assignment.function)

    if kind in {"velocity", "v"}:
        return PrescribedFlowBoundary(
            lambda t, fn=assignment.function, area0=vessel.area0: area0 * fn(t)
        )

    if kind in {"area", "a"}:
        return PrescribedAreaBoundary(assignment.function)

    if kind in {"pressure", "p"}:
        return PrescribedPressureBoundary(assignment.function)

    raise ValueError(f"Unsupported boundary kind {assignment.kind!r}.")


__all__ = [
    "BoundaryAssignment",
    "ScalarFunction",
    "make_boundary_condition",
    "normalize_boundary_kind",
    "role_labels",
]
