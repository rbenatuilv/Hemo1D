from __future__ import annotations

import numpy as np
import pytest

from hemo1d.boundary import CopyBoundaryCondition
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.observe import ProbePoint
from hemo1d.solvers.dg import DGFEMDiscretization, DGMeshConfig, create_dg_vessel
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.time import TimeConfig
from hemo1d.topology import NetworkEndpoint, VascularNetwork


def make_physics(
    *,
    length: float = 1.0,
    area0: float = 0.126,
    beta: float = 0.060606e7,
) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=beta),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_dg_vessel(*, vessel_id: str, physics: Hemo1DPhysics):
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )
    vessel = create_dg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=discretization,
        time_scheme="rk2",
    )
    vessel.interpolate_rest_state()
    return vessel


def test_single_vessel_dg_network_rest_state_stays_rest() -> None:
    physics = make_physics(
        length=1.0,
        area0=0.126,
        beta=0.060606e7,
    )

    vessel = make_dg_vessel(vessel_id="v0", physics=physics)

    network = VascularNetwork(
        vessels={"v0": vessel},
        external_boundaries={
            NetworkEndpoint("v0", EndpointSide.LEFT): CopyBoundaryCondition(),
            NetworkEndpoint("v0", EndpointSide.RIGHT): CopyBoundaryCondition(),
        },
    )

    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=1.0e-4,
            fixed_dt=1.0e-5,
            cfl=0.1,
        ),
        record_every=1,
        probes=[ProbePoint("v0", "center", 0.5)],
        show_progress=False,
    )

    assert result.num_steps == 10
    assert result.time == pytest.approx(1.0e-4)

    z, A, Q = vessel.state_arrays()

    assert z.shape == (16,)
    np.testing.assert_allclose(A, physics.params.area0, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(Q, 0.0, rtol=1.0e-12, atol=1.0e-12)

    assert len(result.history.diagnostics) == 11
    assert len(result.history.probes.samples) == 11


def test_single_vessel_dg_network_with_small_fixed_dt_and_copy_boundaries_runs() -> None:
    physics = make_physics(
        length=1.0,
        area0=0.126,
        beta=0.060606e7,
    )

    vessel = make_dg_vessel(vessel_id="v0", physics=physics)

    # Small smooth perturbation.
    x = vessel.discretization.coordinates()
    vessel.state_n.A[:, :] = physics.params.area0 * (
        1.0 + 0.001 * np.sin(2.0 * np.pi * x)
    )
    vessel.state_n.Q[:, :] = 0.0005 * np.sin(2.0 * np.pi * x)

    network = VascularNetwork(
        vessels={"v0": vessel},
        external_boundaries={
            NetworkEndpoint("v0", EndpointSide.LEFT): CopyBoundaryCondition(),
            NetworkEndpoint("v0", EndpointSide.RIGHT): CopyBoundaryCondition(),
        },
    )

    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=1.0e-5,
            fixed_dt=1.0e-6,
            cfl=0.1,
        ),
        record_every=1,
        probes=[ProbePoint("v0", "center", 0.5)],
        show_progress=False,
    )

    assert result.num_steps == 10

    _, A, Q = vessel.state_arrays()

    assert np.all(A > 0.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(Q))
