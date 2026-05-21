from __future__ import annotations

from hemo1d.boundary.base import BoundaryCondition
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide


class CopyBoundaryCondition(BoundaryCondition):
    """
    Default/fallback boundary condition that copies endpoint values.

    This is used as a placeholder for single-vessel problems where boundary
    conditions are not explicitly provided. For network simulations, actual
    boundary conditions should be specified for inlets and outlets.

    The condition simply extracts and returns the current endpoint state,
    effectively enforcing no flux change at the boundary (homogeneous Dirichlet).
    """

    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        return BoundaryState(
            area=endpoint_data.state.area,
            flow_rate=endpoint_data.state.flow_rate,
        )