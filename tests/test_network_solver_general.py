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
from hemo1d.topology import (
    Bifurcation,
    NetworkEndpoint,
    VascularNetwork,
)
from hemo1d.observe import ProbePoint
from hemo1d.solvers.time import TimeConfig


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_vessel(vessel_id: str, num_cells: int = 16):
    physics = make_physics()
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


def test_general_network_solver_preserves_single_vessel_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network solver tests are serial-only for now.")

    vessel = make_vessel("vessel", num_cells=16)

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
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
    assert result.num_steps == 3

    final_vessel = result.network.vessels["vessel"]

    assert np.allclose(final_vessel.state_n.A.x.array, final_vessel.physics.params.area0)
    assert np.allclose(final_vessel.state_n.Q.x.array, 0.0)


def test_general_network_solver_preserves_three_vessel_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network solver tests are serial-only for now.")

    parent = make_vessel("parent", num_cells=16)
    daughter1 = make_vessel("daughter1", num_cells=16)
    daughter2 = make_vessel("daughter2", num_cells=16)

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
            NetworkEndpoint("parent", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
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
        assert np.allclose(vessel.state_n.A.x.array, vessel.physics.params.area0)
        assert np.allclose(vessel.state_n.Q.x.array, 0.0)


def test_general_network_solver_records_probes():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network solver tests are serial-only for now.")

    vessel = make_vessel("vessel", num_cells=16)

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=2.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
        probes=[
            ProbePoint(vessel_id="vessel", name="left", coordinate=0.0),
            ProbePoint(vessel_id="vessel", name="right", coordinate=vessel.length),
        ],
    )

    assert len(result.history.probes.samples) == 3 * 2
    assert ("vessel", "left") in result.history.probes.keys()
    assert ("vessel", "right") in result.history.probes.keys()


def test_general_network_solver_records_snapshots_only_when_requested():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network solver tests are serial-only for now.")

    vessel = make_vessel("vessel", num_cells=16)

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    solver = NetworkSolver(network)

    result_without_snapshots = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=2.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
    )

    assert result_without_snapshots.history.snapshots.snapshots == []

    vessel = make_vessel("vessel", num_cells=16)
    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )
    solver = NetworkSolver(network)

    sample_points = np.linspace(0.0, vessel.length, 5)
    result_with_snapshots = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=3.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=2,
        snapshot_sample_points_by_vessel={"vessel": sample_points},
    )

    diagnostic_times = result_with_snapshots.history.times
    snapshot_times = result_with_snapshots.history.snapshots.times

    assert snapshot_times == diagnostic_times
    assert snapshot_times == pytest.approx([0.0, 2.0e-5, 3.0e-5])

    for snapshot in result_with_snapshots.history.snapshots.snapshots:
        vessel_snapshot = snapshot.vessel_snapshots["vessel"]
        assert np.allclose(vessel_snapshot.z, sample_points)
        assert np.allclose(vessel_snapshot.area, vessel.physics.params.area0)
        assert np.allclose(vessel_snapshot.flow_rate, 0.0)
