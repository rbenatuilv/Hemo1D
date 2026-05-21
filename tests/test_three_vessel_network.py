import math

import numpy as np
import pytest
from mpi4py import MPI

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.vessel import Vessel
from hemo1d.topology import (
    Bifurcation,
    NetworkEndpoint,
    VascularNetwork,
)
from hemo1d.solvers.time import TimeConfig


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(
            length=length,
            area0=area0,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_vessel(vessel_id: str, physics: Hemo1DPhysics, num_cells: int = 16) -> Vessel:
    disc = CGFEMDiscretization(
        CGMeshConfig(
            length=physics.params.length,
            num_cells=num_cells,
            degree=1,
        )
    )

    vessel = create_cg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=disc,
    )
    vessel.interpolate_rest_state()

    return vessel


def make_rest_network(num_cells: int = 16) -> VascularNetwork:
    parent_physics = make_physics(length=2.0, area0=0.126)
    d1_physics = make_physics(length=2.0, area0=0.126)
    d2_physics = make_physics(length=2.0, area0=0.126)

    parent = make_vessel("parent", parent_physics, num_cells=num_cells)
    daughter1 = make_vessel("daughter1", d1_physics, num_cells=num_cells)
    daughter2 = make_vessel("daughter2", d2_physics, num_cells=num_cells)

    bifurcation = Bifurcation(
        parent=NetworkEndpoint("parent", EndpointSide.RIGHT),
        daughter1=NetworkEndpoint("daughter1", EndpointSide.LEFT),
        daughter2=NetworkEndpoint("daughter2", EndpointSide.LEFT),
    )

    return VascularNetwork(
        vessels={
            "parent": parent,
            "daughter1": daughter1,
            "daughter2": daughter2,
        },
        bifurcations=[bifurcation],
        external_boundaries={
            NetworkEndpoint("parent", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("daughter1", EndpointSide.RIGHT): NonReflectingBoundary(),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )


def test_vessel_create_and_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network tests are serial-only for now.")

    physics = make_physics()
    vessel = make_vessel("v", physics, num_cells=8)

    assert vessel.vessel_id == "v"
    assert np.allclose(vessel.state_n.A.x.array, physics.params.area0)
    assert np.allclose(vessel.state_n.Q.x.array, 0.0)


def test_three_vessel_network_preserves_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network tests are serial-only for now.")

    network = make_rest_network(num_cells=16)
    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=3.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
    )

    assert math.isclose(result.time, 3.0e-5)
    assert result.num_steps == 3
    assert len(result.history.diagnostics) == 4

    for vessel in result.network.vessels.values():
        assert np.allclose(
            vessel.state_n.A.x.array,
            vessel.physics.params.area0,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        assert np.allclose(
            vessel.state_n.Q.x.array,
            0.0,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_three_vessel_network_small_inlet_pulse_remains_finite():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network tests are serial-only for now.")

    parent_physics = make_physics(length=2.0, area0=0.126)
    d1_physics = make_physics(length=2.0, area0=0.126)
    d2_physics = make_physics(length=2.0, area0=0.126)

    parent = make_vessel("parent", parent_physics, num_cells=32)
    daughter1 = make_vessel("daughter1", d1_physics, num_cells=32)
    daughter2 = make_vessel("daughter2", d2_physics, num_cells=32)

    def q_in(t: float) -> float:
        T = 1.0e-4
        amp = 1.0e-4
        return amp * np.sin(np.pi * t / T) if 0.0 <= t <= T else 0.0

    bifurcation = Bifurcation(
        parent=NetworkEndpoint("parent", EndpointSide.RIGHT),
        daughter1=NetworkEndpoint("daughter1", EndpointSide.LEFT),
        daughter2=NetworkEndpoint("daughter2", EndpointSide.LEFT),
    )

    network = VascularNetwork(
        vessels={
            "parent": parent,
            "daughter1": daughter1,
            "daughter2": daughter2,
        },
        bifurcations=[bifurcation],
        external_boundaries={
            NetworkEndpoint("parent", EndpointSide.LEFT): PrescribedFlowBoundary(q_in),
            NetworkEndpoint("daughter1", EndpointSide.RIGHT): NonReflectingBoundary(),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=3.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
    )

    assert math.isclose(result.time, 3.0e-5)

    for vessel in result.network.vessels.values():
        assert np.all(np.isfinite(vessel.state_n.A.x.array))
        assert np.all(np.isfinite(vessel.state_n.Q.x.array))
        assert np.min(vessel.state_n.A.x.array) > 0.0


def test_network_compute_dt_uses_minimum_vessel_dt():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network tests are serial-only for now.")

    network = make_rest_network(num_cells=16)
    solver = NetworkSolver(network)

    config = TimeConfig(
        t0=0.0,
        t_end=1.0e-3,
        fixed_dt=None,
        cfl=0.25,
    )

    dt = solver.compute_dt(time=0.0, config=config)

    assert dt > 0.0
    assert dt <= config.t_end
