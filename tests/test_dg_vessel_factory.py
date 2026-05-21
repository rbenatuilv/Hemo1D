from __future__ import annotations

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.convergence.network_snapshots import NetworkSnapshotRecorder
from hemo1d.solvers.dg import (
    DGFEMDiscretization,
    DGMeshConfig,
    create_dg_vessel,
    extract_dg_state_arrays,
    sample_dg_state,
    sample_dg_state_array,
)
from hemo1d.solvers.vessel import Vessel


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(
            length=1.0,
            area0=0.126,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_sample_dg_state_degree_one_exact_for_linear_function(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")

    discretization.interpolate_state(
        state,
        area_fn=lambda x: 1.0 + 2.0 * x,
        flow_rate_fn=lambda x: -1.0 + 3.0 * x,
    )

    coordinate = 0.375

    sampled = sample_dg_state(
        discretization=discretization,
        state=state,
        physics=physics,
        coordinate=coordinate,
    )

    assert sampled.area == pytest.approx(1.0 + 2.0 * coordinate)
    assert sampled.flow_rate == pytest.approx(-1.0 + 3.0 * coordinate)


def test_sample_dg_state_at_right_endpoint(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")

    discretization.interpolate_state(
        state,
        area_fn=lambda x: 1.0 + x,
        flow_rate_fn=lambda x: 2.0 - x,
    )

    sampled = sample_dg_state(
        discretization=discretization,
        state=state,
        physics=physics,
        coordinate=1.0,
    )

    assert sampled.area == pytest.approx(2.0)
    assert sampled.flow_rate == pytest.approx(1.0)


@pytest.mark.parametrize("degree", [0, 1])
def test_sample_dg_state_array_matches_scalar_sampling(
    physics: Hemo1DPhysics,
    degree: int,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=degree)
    )
    state = discretization.create_state(name="n")

    x = discretization.coordinates()
    state.A[:, :] = 1.0 + x
    state.Q[:, :] = 2.0 - x

    coordinates = np.array(
        [
            0.0,
            0.125,
            0.25,
            0.5,
            0.875,
            1.0,
        ],
        dtype=float,
    )

    areas, flows = sample_dg_state_array(
        discretization=discretization,
        state=state,
        physics=physics,
        coordinates=coordinates,
    )

    scalar_samples = [
        sample_dg_state(
            discretization=discretization,
            state=state,
            physics=physics,
            coordinate=float(coordinate),
        )
        for coordinate in coordinates
    ]

    np.testing.assert_allclose(
        areas,
        [sample.area for sample in scalar_samples],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        flows,
        [sample.flow_rate for sample in scalar_samples],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_dg_vectorized_sampler_uses_right_cell_at_interior_interface(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")

    state.A[:, 0] = [1.0, 10.0, 20.0, 30.0]
    state.A[:, 1] = [2.0, 11.0, 21.0, 31.0]
    state.Q[:, 0] = [-1.0, -10.0, -20.0, -30.0]
    state.Q[:, 1] = [-2.0, -11.0, -21.0, -31.0]

    areas, flows = sample_dg_state_array(
        discretization=discretization,
        state=state,
        physics=physics,
        coordinates=np.array([0.25, 1.0]),
    )

    assert areas[0] == pytest.approx(state.A[1, 0])
    assert flows[0] == pytest.approx(state.Q[1, 0])
    assert areas[1] == pytest.approx(state.A[-1, -1])
    assert flows[1] == pytest.approx(state.Q[-1, -1])


def test_network_snapshot_recorder_uses_dg_bulk_sampling_values(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    vessel = create_dg_vessel(
        vessel_id="v0",
        physics=physics,
        discretization=discretization,
    )

    x = discretization.coordinates()
    vessel.state_n.A[:, :] = 1.0 + x
    vessel.state_n.Q[:, :] = 2.0 - x

    points = np.array([0.0, 0.1, 0.25, 0.7, 1.0])
    recorder = NetworkSnapshotRecorder(
        vessels={"v0": vessel},
        sample_points_by_vessel={"v0": points},
    )

    snapshot = recorder.sample(time=0.0).vessel_snapshots["v0"]
    scalar_samples = [vessel.sample_state(float(point)) for point in points]

    np.testing.assert_allclose(
        snapshot.area,
        [sample.area for sample in scalar_samples],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        snapshot.flow_rate,
        [sample.flow_rate for sample in scalar_samples],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_extract_dg_state_arrays_degree_one_has_duplicate_interface_nodes() -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")

    x = discretization.coordinates()
    state.A[:, :] = 1.0 + x
    state.Q[:, :] = 2.0 - x

    z, A, Q = extract_dg_state_arrays(discretization, state)

    assert z.shape == (8,)
    assert A.shape == (8,)
    assert Q.shape == (8,)

    np.testing.assert_allclose(A, 1.0 + z)
    np.testing.assert_allclose(Q, 2.0 - z)

    # Interface x = 0.25 appears twice for degree-1 DG:
    # right node of cell 0 and left node of cell 1.
    assert z[1] == pytest.approx(0.25)
    assert z[2] == pytest.approx(0.25)


def test_create_dg_vessel_returns_generic_vessel(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )

    vessel = create_dg_vessel(
        vessel_id="v0",
        physics=physics,
        discretization=discretization,
        time_scheme="rk2",
    )

    assert isinstance(vessel, Vessel)
    assert vessel.vessel_id == "v0"
    assert vessel.length == pytest.approx(1.0)
    assert vessel.num_dofs == 8

    vessel.interpolate_rest_state()

    sampled = vessel.sample_state(0.5)

    assert sampled.area == pytest.approx(physics.params.area0)
    assert sampled.flow_rate == pytest.approx(0.0)


def test_vessel_state_arrays_use_dg_extractor(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )

    vessel = create_dg_vessel(
        vessel_id="v0",
        physics=physics,
        discretization=discretization,
    )
    vessel.interpolate_rest_state()

    z, A, Q = vessel.state_arrays()

    assert z.shape == (8,)
    assert A.shape == (8,)
    assert Q.shape == (8,)
    np.testing.assert_allclose(A, physics.params.area0)
    np.testing.assert_allclose(Q, 0.0)
