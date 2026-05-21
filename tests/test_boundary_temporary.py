from hemo1d.boundary import CopyBoundaryCondition
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint


def make_physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_copy_boundary_returns_same_state_on_left():
    physics = make_physics()
    bc = CopyBoundaryCondition()

    endpoint_data = EndpointData(
        state=StateAtPoint(area=0.15, flow_rate=0.20),
        d_area_dz=1.0,
        d_flow_rate_dz=-2.0,
    )

    boundary = bc.compute(
        physics=physics,
        endpoint_data=endpoint_data,
        side=EndpointSide.LEFT,
        t=0.01,
        dt=1.0e-5,
    )

    assert boundary.area == endpoint_data.state.area
    assert boundary.flow_rate == endpoint_data.state.flow_rate


def test_copy_boundary_returns_same_state_on_right():
    physics = make_physics()
    bc = CopyBoundaryCondition()

    endpoint_data = EndpointData(
        state=StateAtPoint(area=0.20, flow_rate=-0.10),
        d_area_dz=0.0,
        d_flow_rate_dz=0.0,
    )

    boundary = bc.compute(
        physics=physics,
        endpoint_data=endpoint_data,
        side=EndpointSide.RIGHT,
        t=0.01,
        dt=1.0e-5,
    )

    assert boundary.area == endpoint_data.state.area
    assert boundary.flow_rate == endpoint_data.state.flow_rate