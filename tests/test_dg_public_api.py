from __future__ import annotations

import pytest

import hemo1d as hd
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.dg import DGFEMDiscretization, DGMeshConfig, create_dg_vessel


def make_physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=1.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_low_level_dg_factory_is_canonical() -> None:
    physics = make_physics()
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )

    vessel = create_dg_vessel(
        vessel_id="dg_vessel",
        physics=physics,
        discretization=discretization,
        time_scheme="rk2",
    )
    vessel.interpolate_rest_state()

    assert vessel.vessel_id == "dg_vessel"
    assert vessel.length == pytest.approx(1.0)
    assert vessel.num_dofs == 8

    sampled = vessel.sample_state(0.5)

    assert sampled.area == pytest.approx(physics.params.area0)
    assert sampled.flow_rate == pytest.approx(0.0)


def test_top_level_public_api_hides_dg_implementation_details() -> None:
    assert hd.load_from_config is not None
    assert not hasattr(hd, "DGFEMDiscretization")
    assert not hasattr(hd, "create_dg_vessel")
