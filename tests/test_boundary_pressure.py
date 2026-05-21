import pytest

from hemo1d.boundary.external import PrescribedPressureBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint


def make_physics():
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=1.0, area0=0.126, beta=0.060606e7),
        p0=10.0,
        p_ext=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_prescribed_pressure_boundary_matches_reference_area_at_reference_pressure():
    physics = make_physics()
    boundary = PrescribedPressureBoundary(lambda t: physics.pressure(physics.params.area0))
    endpoint_data = EndpointData(
        state=StateAtPoint(area=physics.params.area0, flow_rate=0.0),
        d_area_dz=0.0,
        d_flow_rate_dz=0.0,
    )

    state = boundary.compute(
        physics=physics,
        endpoint_data=endpoint_data,
        side=EndpointSide.LEFT,
        t=0.0,
        dt=1.0e-5,
    )

    assert state.area == pytest.approx(physics.params.area0)
    assert state.flow_rate == pytest.approx(0.0)
