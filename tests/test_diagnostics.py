import math

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.observe.diagnostics import compute_cg_state_diagnostics
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig


def make_physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_diagnostics_at_rest():
    physics = make_physics()

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )
    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    diagnostics = compute_cg_state_diagnostics(state, physics, time=0.25)

    assert math.isclose(diagnostics.time, 0.25)
    assert math.isclose(diagnostics.min_area, physics.params.area0)
    assert math.isclose(diagnostics.max_area, physics.params.area0)
    assert math.isclose(diagnostics.min_flow_rate, 0.0)
    assert math.isclose(diagnostics.max_flow_rate, 0.0)
    assert math.isclose(diagnostics.max_pressure, physics.pressure(physics.params.area0))
    assert diagnostics.max_wave_speed > 0.0