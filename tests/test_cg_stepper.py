import numpy as np
import pytest
from mpi4py import MPI

from hemo1d.boundary import CopyBoundaryCondition
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointSide
from hemo1d.solvers.cg import (
    CGFEMDiscretization,
    CGMeshConfig,
    CGTaylorGalerkinStepper,
)


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_compute_stable_dt_is_positive(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    dt = stepper.compute_stable_dt(state)

    assert dt > 0.0


def test_compute_stable_dt_rejects_bad_cfl(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    with pytest.raises(ValueError):
        stepper.compute_stable_dt(state, cfl=0.0)


def test_compute_boundary_states_with_copy_boundary(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    left_state, right_state = stepper.compute_boundary_states(
        state_n=state,
        t_np1=1.0e-5,
        dt=1.0e-5,
        left_boundary=CopyBoundaryCondition(),
        right_boundary=CopyBoundaryCondition(),
    )

    assert left_state.area == physics.params.area0
    assert left_state.flow_rate == 0.0
    assert right_state.area == physics.params.area0
    assert right_state.flow_rate == 0.0


def test_one_step_preserves_rest_state_with_copy_boundaries(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=32, degree=1)
    )

    state_n = disc.create_state(name="n")
    state_np1 = disc.create_state(name="np1")

    disc.interpolate_rest_state(state_n, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    dt = min(1.0e-5, 0.25 * stepper.compute_stable_dt(state_n))

    stepper.step_with_boundary_conditions(
        state_n=state_n,
        state_np1=state_np1,
        t_n=0.0,
        dt=dt,
        left_boundary=CopyBoundaryCondition(),
        right_boundary=CopyBoundaryCondition(),
    )

    assert np.allclose(state_np1.A.x.array, physics.params.area0, rtol=1e-11, atol=1e-11)
    assert np.allclose(state_np1.Q.x.array, 0.0, rtol=1e-11, atol=1e-11)


def test_one_step_with_manual_boundary_states_preserves_rest_state(
    physics: Hemo1DPhysics,
):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=32, degree=1)
    )

    state_n = disc.create_state(name="n")
    state_np1 = disc.create_state(name="np1")

    disc.interpolate_rest_state(state_n, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    dt = min(1.0e-5, 0.25 * stepper.compute_stable_dt(state_n))

    rest_boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper.step(
        state_n=state_n,
        state_np1=state_np1,
        dt=dt,
        left_boundary_state=rest_boundary,
        right_boundary_state=rest_boundary,
    )

    assert np.allclose(state_np1.A.x.array, physics.params.area0, rtol=1e-11, atol=1e-11)
    assert np.allclose(state_np1.Q.x.array, 0.0, rtol=1e-11, atol=1e-11)


def test_one_step_from_smooth_area_perturbation_is_finite(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=32, degree=1)
    )

    state_n = disc.create_state(name="n")
    state_np1 = disc.create_state(name="np1")

    A0 = physics.params.area0
    L = physics.params.length

    state_n.A.interpolate(lambda x: A0 * (1.0 + 1.0e-3 * np.sin(np.pi * x[0] / L)))
    state_n.Q.interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
    state_n.scatter_forward()

    stepper = CGTaylorGalerkinStepper(disc, physics)

    dt = 0.25 * stepper.compute_stable_dt(state_n)

    left_endpoint = disc.endpoint_state(state_n, EndpointSide.LEFT)
    right_endpoint = disc.endpoint_state(state_n, EndpointSide.RIGHT)

    stepper.step(
        state_n=state_n,
        state_np1=state_np1,
        dt=dt,
        left_boundary_state=BoundaryState(
            area=left_endpoint.area,
            flow_rate=left_endpoint.flow_rate,
        ),
        right_boundary_state=BoundaryState(
            area=right_endpoint.area,
            flow_rate=right_endpoint.flow_rate,
        ),
    )

    assert np.all(np.isfinite(state_np1.A.x.array))
    assert np.all(np.isfinite(state_np1.Q.x.array))
    assert np.min(state_np1.A.x.array) > 0.0


def test_step_rejects_non_positive_dt(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Stepper currently serial-oriented.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    state_n = disc.create_state(name="n")
    state_np1 = disc.create_state(name="np1")

    disc.interpolate_rest_state(state_n, physics)

    stepper = CGTaylorGalerkinStepper(disc, physics)

    rest_boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    with pytest.raises(ValueError):
        stepper.step(
            state_n=state_n,
            state_np1=state_np1,
            dt=0.0,
            left_boundary_state=rest_boundary,
            right_boundary_state=rest_boundary,
        )