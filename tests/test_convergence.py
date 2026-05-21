import math

import numpy as np
import pytest
from mpi4py import MPI

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.convergence import (
    SnapshotHistory,
    SolutionSnapshot,
    VesselSnapshotRecorder,
    l2_error_1d,
    linf_time_l2_space_error,
    observed_orders,
    richardson_extrapolate_history,
    richardson_extrapolate_values,
)
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.topology import NetworkEndpoint, VascularNetwork
from hemo1d.solvers.time import TimeConfig


def test_l2_error_1d_is_zero_for_identical_arrays():
    z = np.linspace(0.0, 1.0, 11)
    u = z**2

    assert math.isclose(l2_error_1d(u, u, z), 0.0)


def test_l2_error_1d_for_constant_difference():
    z = np.linspace(0.0, 2.0, 101)
    u = np.ones_like(z)
    v = np.zeros_like(z)

    assert math.isclose(l2_error_1d(u, v, z), math.sqrt(2.0), rel_tol=1.0e-4)


def test_richardson_extrapolate_values_for_second_order_model_error():
    exact = np.array([1.0, 2.0, 3.0])
    h = 0.1
    c = 5.0

    coarse = exact + c * h**2
    fine = exact + c * (h / 2.0) ** 2

    extrapolated = richardson_extrapolate_values(
        coarse_values=coarse,
        fine_values=fine,
        expected_order=2.0,
        delta=2.0,
    )

    assert np.allclose(extrapolated, exact, rtol=1.0e-14, atol=1.0e-14)


def test_observed_orders_for_halving_errors():
    errors = [1.0, 0.25, 0.0625]

    orders = observed_orders(errors, delta=2.0)

    assert np.allclose(orders, [2.0, 2.0])


def test_linf_time_l2_space_error():
    z = np.linspace(0.0, 1.0, 11)

    h1 = SnapshotHistory(
        snapshots=[
            SolutionSnapshot(
                time=0.0,
                z=z,
                area=z,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            ),
            SolutionSnapshot(
                time=1.0,
                z=z,
                area=z + 1.0,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            ),
        ]
    )

    h2 = SnapshotHistory(
        snapshots=[
            SolutionSnapshot(
                time=0.0,
                z=z,
                area=z,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            ),
            SolutionSnapshot(
                time=1.0,
                z=z,
                area=z,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            ),
        ]
    )

    error = linf_time_l2_space_error(h1, h2, field="area")

    assert math.isclose(error, 1.0, rel_tol=1.0e-4)


def test_richardson_extrapolate_history():
    z = np.linspace(0.0, 1.0, 11)
    exact_area = 1.0 + z

    h = 0.1
    c = 3.0

    coarse = SnapshotHistory(
        snapshots=[
            SolutionSnapshot(
                time=0.0,
                z=z,
                area=exact_area + c * h**2,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            )
        ]
    )

    fine = SnapshotHistory(
        snapshots=[
            SolutionSnapshot(
                time=0.0,
                z=z,
                area=exact_area + c * (h / 2.0) ** 2,
                flow_rate=z * 0.0,
                pressure=z * 0.0,
            )
        ]
    )

    rich = richardson_extrapolate_history(
        coarse=coarse,
        fine=fine,
        expected_order=2.0,
        delta=2.0,
    )

    assert np.allclose(rich.snapshots[0].area, exact_area, rtol=1.0e-14, atol=1.0e-14)


def test_network_solver_records_single_vessel_snapshots():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Snapshot recorder is serial-only for now.")

    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
    )
    physics = Hemo1DPhysics(params, NP_BACKEND)

    disc = CGFEMDiscretization(
        CGMeshConfig(length=params.length, num_cells=8, degree=1)
    )

    vessel = create_cg_vessel(
        vessel_id="vessel",
        physics=physics,
        discretization=disc,
    )
    vessel.interpolate_rest_state()

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    solver = NetworkSolver(network)

    snapshot_points = np.linspace(0.0, params.length, 17)
    recorder = VesselSnapshotRecorder(
        vessel=vessel,
        sample_points=snapshot_points,
    )

    history = SnapshotHistory()
    time = 0.0
    history.snapshots.append(recorder.sample(time=time))

    config = TimeConfig(t0=0.0, t_end=2.0e-5, fixed_dt=1.0e-5)

    for _ in range(2):
        dt = solver.compute_dt(time=time, config=config)
        solver.step(time=time, dt=dt)
        time += dt
        history.snapshots.append(recorder.sample(time=time))

    assert len(history.snapshots) == 3

    for snapshot in history.snapshots:
        assert np.allclose(snapshot.z, snapshot_points)
        assert np.allclose(snapshot.area, params.area0)
        assert np.allclose(snapshot.flow_rate, 0.0)
