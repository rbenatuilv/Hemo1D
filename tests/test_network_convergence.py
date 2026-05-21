import math

import numpy as np
import pytest
from mpi4py import MPI

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.convergence import (
    ConvergenceLevel,
    NetworkSnapshotHistory,
    NetworkSnapshotRecorder,
    NetworkSolutionSnapshot,
    compute_network_richardson_error_rows,
    linf_time_network_l2_error,
    network_l2_error_at_time,
    network_observed_orders,
    richardson_extrapolate_network_history,
)
from hemo1d.convergence.snapshots import SolutionSnapshot
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.solvers.vessel import Vessel
from hemo1d.topology import (
    Bifurcation,
    NetworkEndpoint,
    VascularNetwork,
)


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=0.060606e7),
        gamma_profile=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_vessel(vessel_id: str, physics: Hemo1DPhysics, num_cells: int = 8) -> Vessel:
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=num_cells, degree=1)
    )
    vessel = create_cg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=disc,
    )
    vessel.interpolate_rest_state()
    return vessel


def make_network(num_cells: int = 8) -> VascularNetwork:
    parent = make_vessel("parent", make_physics(), num_cells=num_cells)
    d1 = make_vessel("daughter1", make_physics(), num_cells=num_cells)
    d2 = make_vessel("daughter2", make_physics(), num_cells=num_cells)

    bifurcation = Bifurcation(
        parent=NetworkEndpoint("parent", EndpointSide.RIGHT),
        daughter1=NetworkEndpoint("daughter1", EndpointSide.LEFT),
        daughter2=NetworkEndpoint("daughter2", EndpointSide.LEFT),
    )

    return VascularNetwork(
        vessels={
            "parent": parent,
            "daughter1": d1,
            "daughter2": d2,
        },
        bifurcations=[bifurcation],
        external_boundaries={
            NetworkEndpoint("parent", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("daughter1", EndpointSide.RIGHT): NonReflectingBoundary(),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )


def make_network_snapshot_history(offset: float = 0.0) -> NetworkSnapshotHistory:
    z = np.linspace(0.0, 1.0, 11)

    def vessel_snapshot(time: float, scale: float) -> SolutionSnapshot:
        area = 1.0 + scale * z + offset
        flow = scale * z * z + offset
        pressure = 2.0 * area

        return SolutionSnapshot(
            time=time,
            z=z,
            area=area,
            flow_rate=flow,
            pressure=pressure,
        )

    history = NetworkSnapshotHistory()

    for time in [0.0, 1.0]:
        history.snapshots.append(
            NetworkSolutionSnapshot(
                time=time,
                vessel_snapshots={
                    "parent": vessel_snapshot(time, scale=1.0),
                    "daughter1": vessel_snapshot(time, scale=2.0),
                    "daughter2": vessel_snapshot(time, scale=3.0),
                },
            )
        )

    return history


def test_network_snapshot_recorder_samples_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network snapshot recorder is serial-only for now.")

    network = make_network(num_cells=8)

    sample_points = np.linspace(0.0, 2.0, 17)

    recorder = NetworkSnapshotRecorder(
        vessels=network.vessels,
        sample_points_by_vessel={
            "parent": sample_points,
            "daughter1": sample_points,
            "daughter2": sample_points,
        },
    )

    snapshot = recorder.sample(time=0.0)

    assert snapshot.time == 0.0
    assert set(snapshot.vessel_snapshots) == {"parent", "daughter1", "daughter2"}

    for vessel_snapshot in snapshot.vessel_snapshots.values():
        assert np.allclose(vessel_snapshot.z, sample_points)
        assert np.allclose(vessel_snapshot.area, 0.126)
        assert np.allclose(vessel_snapshot.flow_rate, 0.0)


def test_network_snapshot_recorder_rejects_missing_sample_points():
    network = make_network(num_cells=8)
    sample_points = np.linspace(0.0, 2.0, 17)

    with pytest.raises(ValueError):
        NetworkSnapshotRecorder(
            vessels=network.vessels,
            sample_points_by_vessel={
                "parent": sample_points,
                "daughter1": sample_points,
            },
        )


def test_network_l2_error_is_zero_for_identical_histories():
    history = make_network_snapshot_history(offset=0.0)

    snapshot = history.snapshots[0]

    error = network_l2_error_at_time(
        solution=snapshot,
        reference=snapshot,
        field="area",
    )

    assert math.isclose(error, 0.0)


def test_network_l2_error_for_constant_offset():
    reference = make_network_snapshot_history(offset=0.0)
    solution = make_network_snapshot_history(offset=1.0)

    error = network_l2_error_at_time(
        solution=solution.snapshots[0],
        reference=reference.snapshots[0],
        field="area",
    )

    assert math.isclose(error, math.sqrt(3.0), rel_tol=1.0e-4)


def test_linf_time_network_l2_error():
    reference = make_network_snapshot_history(offset=0.0)
    solution = make_network_snapshot_history(offset=1.0)

    error = linf_time_network_l2_error(
        solution=solution,
        reference=reference,
        field="area",
    )

    assert math.isclose(error, math.sqrt(3.0), rel_tol=1.0e-4)


def test_richardson_extrapolate_network_history_second_order_error():
    z = np.linspace(0.0, 1.0, 11)

    exact_area = 1.0 + z
    exact_flow = 2.0 + z
    exact_pressure = 3.0 + z

    h = 0.1
    c = 5.0

    def history_with_error(error_size: float) -> NetworkSnapshotHistory:
        history = NetworkSnapshotHistory()
        history.snapshots.append(
            NetworkSolutionSnapshot(
                time=0.0,
                vessel_snapshots={
                    vessel_id: SolutionSnapshot(
                        time=0.0,
                        z=z,
                        area=exact_area + error_size,
                        flow_rate=exact_flow + error_size,
                        pressure=exact_pressure + error_size,
                    )
                    for vessel_id in ["parent", "daughter1", "daughter2"]
                },
            )
        )
        return history

    coarse = history_with_error(c * h**2)
    fine = history_with_error(c * (h / 2.0) ** 2)

    rich = richardson_extrapolate_network_history(
        coarse=coarse,
        fine=fine,
        expected_order=2.0,
        delta=2.0,
    )

    for snapshot in rich.snapshots[0].vessel_snapshots.values():
        assert np.allclose(snapshot.area, exact_area)
        assert np.allclose(snapshot.flow_rate, exact_flow)
        assert np.allclose(snapshot.pressure, exact_pressure)


def test_compute_network_richardson_error_rows():
    z = np.linspace(0.0, 1.0, 11)

    exact_area = 1.0 + z
    exact_flow = 2.0 + z

    def make_history(error_size: float) -> NetworkSnapshotHistory:
        history = NetworkSnapshotHistory()
        history.snapshots.append(
            NetworkSolutionSnapshot(
                time=0.0,
                vessel_snapshots={
                    vessel_id: SolutionSnapshot(
                        time=0.0,
                        z=z,
                        area=exact_area + error_size,
                        flow_rate=exact_flow + error_size,
                        pressure=exact_area * 0.0,
                    )
                    for vessel_id in ["parent", "daughter1", "daughter2"]
                },
            )
        )
        return history

    levels = [
        ConvergenceLevel(name="N16", num_cells=16, dt=1.0e-4),
        ConvergenceLevel(name="N32", num_cells=32, dt=5.0e-5),
        ConvergenceLevel(name="N64", num_cells=64, dt=2.5e-5),
    ]

    histories = {
        "N16": make_history(1.0e-2),
        "N32": make_history(2.5e-3),
        "N64": make_history(6.25e-4),
    }

    rows = compute_network_richardson_error_rows(
        level_histories=histories,
        levels=levels,
        expected_order=2.0,
        delta=2.0,
    )

    assert len(rows) == 2
    assert rows[0].level_name == "N16"
    assert rows[1].level_name == "N32"

    area_orders, flow_orders = network_observed_orders(rows)

    assert len(area_orders) == 1
    assert len(flow_orders) == 1
