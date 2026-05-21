import math

import numpy as np

from hemo1d.boundary.external import (
    NonReflectingBoundary,
    PrescribedAreaBoundary,
    PrescribedFlowBoundary,
)
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint


def make_physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def rest_endpoint_data(physics: Hemo1DPhysics) -> EndpointData:
    return EndpointData(
        state=StateAtPoint(area=physics.params.area0, flow_rate=0.0),
        d_area_dz=0.0,
        d_flow_rate_dz=0.0,
    )


def test_prescribed_flow_boundary_at_rest_preserves_area_on_left():
    physics = make_physics()
    data = rest_endpoint_data(physics)

    bc = PrescribedFlowBoundary(flow_rate=lambda t: 0.0)

    state = bc.compute(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.LEFT,
        t=1.0e-5,
        dt=1.0e-5,
    )

    assert math.isclose(state.area, physics.params.area0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(state.flow_rate, 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_prescribed_area_boundary_at_rest_preserves_flow_on_left():
    physics = make_physics()
    data = rest_endpoint_data(physics)

    bc = PrescribedAreaBoundary(area=lambda t: physics.params.area0)

    state = bc.compute(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.LEFT,
        t=1.0e-5,
        dt=1.0e-5,
    )

    assert math.isclose(state.area, physics.params.area0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(state.flow_rate, 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_non_reflecting_boundary_at_rest_preserves_state_on_right():
    physics = make_physics()
    data = rest_endpoint_data(physics)

    bc = NonReflectingBoundary()

    state = bc.compute(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.RIGHT,
        t=1.0e-5,
        dt=1.0e-5,
    )

    assert math.isclose(state.area, physics.params.area0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(state.flow_rate, 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_non_reflecting_boundary_returns_finite_values_for_small_perturbation():
    physics = make_physics()

    data = EndpointData(
        state=StateAtPoint(area=1.001 * physics.params.area0, flow_rate=1.0e-3),
        d_area_dz=1.0e-4,
        d_flow_rate_dz=-1.0e-4,
    )

    bc = NonReflectingBoundary()

    state = bc.compute(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.RIGHT,
        t=1.0e-5,
        dt=1.0e-5,
    )

    assert np.isfinite(state.area)
    assert np.isfinite(state.flow_rate)
    assert state.area > 0.0
