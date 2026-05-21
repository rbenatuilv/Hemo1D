from __future__ import annotations

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState
from hemo1d.solvers.dg import (
    DGFEMDiscretization,
    DGLaxFriedrichsStepper,
    DGMeshConfig,
)
from hemo1d.solvers.dg.residual import compute_interface_fluxes


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


@pytest.fixture
def discretization() -> DGFEMDiscretization:
    return DGFEMDiscretization(
        DGMeshConfig(
            length=1.0,
            num_cells=8,
            degree=1,
        )
    )


def test_compute_stable_dt_is_positive(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state = discretization.create_state(name="n")
    discretization.interpolate_rest_state(state, physics)

    stepper = DGLaxFriedrichsStepper(discretization, physics)
    dt = stepper.compute_stable_dt(state, cfl=0.25)

    assert dt > 0.0
    assert np.isfinite(dt)


def test_residual_is_zero_for_rest_state(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state = discretization.create_state(name="n")
    discretization.interpolate_rest_state(state, physics)

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(discretization, physics)

    residual_A, residual_Q = stepper.compute_residual(
        state=state,
        left_boundary_state=boundary,
        right_boundary_state=boundary,
    )

    np.testing.assert_allclose(residual_A, 0.0, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(residual_Q, 0.0, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("degree", [0, 1])
def test_optimized_residual_matches_einsum_reference(
    physics: Hemo1DPhysics,
    degree: int,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(
            length=1.0,
            num_cells=7,
            degree=degree,
        )
    )
    state = discretization.create_state(name="n")

    rng = np.random.default_rng(1234 + degree)
    state.A[:, :] = physics.params.area0 * (1.0 + 0.01 * rng.random(state.A.shape))
    state.Q[:, :] = 1.0e-4 * rng.normal(size=state.Q.shape)

    left_boundary = BoundaryState(area=physics.params.area0 * 1.01, flow_rate=2.0e-5)
    right_boundary = BoundaryState(area=physics.params.area0 * 0.99, flow_rate=-1.0e-5)

    stepper = DGLaxFriedrichsStepper(discretization, physics)

    residual_A, residual_Q = stepper.compute_residual(
        state=state,
        left_boundary_state=left_boundary,
        right_boundary_state=right_boundary,
    )
    reference_A, reference_Q = _compute_residual_einsum_reference(
        physics=physics,
        stepper=stepper,
        state=state,
        left_boundary_state=left_boundary,
        right_boundary_state=right_boundary,
    )

    np.testing.assert_allclose(residual_A, reference_A, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(residual_Q, reference_Q, rtol=1.0e-14, atol=1.0e-14)


def test_rhs_is_zero_for_rest_state(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state = discretization.create_state(name="n")
    discretization.interpolate_rest_state(state, physics)

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(discretization, physics)
    rhs = stepper.compute_rhs(
        state=state,
        left_boundary_state=boundary,
        right_boundary_state=boundary,
    )

    np.testing.assert_allclose(rhs.dA_dt, 0.0, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(rhs.dQ_dt, 0.0, rtol=1.0e-12, atol=1.0e-12)


def test_rest_state_remains_rest_after_euler_step(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state_n = discretization.create_state(name="n")
    state_np1 = discretization.create_state(name="np1")
    discretization.interpolate_rest_state(state_n, physics)

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(
        discretization,
        physics,
        time_scheme="euler",
    )
    dt = stepper.compute_stable_dt(state_n, cfl=0.25)

    stepper.step(
        state_n=state_n,
        state_np1=state_np1,
        dt=dt,
        left_boundary_state=boundary,
        right_boundary_state=boundary,
    )

    np.testing.assert_allclose(state_np1.A, state_n.A, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(state_np1.Q, state_n.Q, rtol=1.0e-12, atol=1.0e-12)


def test_rest_state_remains_rest_after_rk2_step(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state_n = discretization.create_state(name="n")
    state_np1 = discretization.create_state(name="np1")
    discretization.interpolate_rest_state(state_n, physics)

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(
        discretization,
        physics,
        time_scheme="rk2",
    )
    dt = stepper.compute_stable_dt(state_n, cfl=0.25)

    stepper.step(
        state_n=state_n,
        state_np1=state_np1,
        dt=dt,
        left_boundary_state=boundary,
        right_boundary_state=boundary,
    )

    np.testing.assert_allclose(state_np1.A, state_n.A, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(state_np1.Q, state_n.Q, rtol=1.0e-12, atol=1.0e-12)


def test_one_euler_step_keeps_area_positive_for_smooth_perturbation(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state_n = discretization.create_state(name="n")
    state_np1 = discretization.create_state(name="np1")

    x = discretization.coordinates()

    state_n.A[:, :] = physics.params.area0 * (
        1.0 + 0.005 * np.sin(2.0 * np.pi * x)
    )
    state_n.Q[:, :] = 0.002 * np.sin(2.0 * np.pi * x)

    left_boundary = BoundaryState(
        area=float(state_n.A[0, 0]),
        flow_rate=float(state_n.Q[0, 0]),
    )
    right_boundary = BoundaryState(
        area=float(state_n.A[-1, -1]),
        flow_rate=float(state_n.Q[-1, -1]),
    )

    stepper = DGLaxFriedrichsStepper(
        discretization,
        physics,
        time_scheme="euler",
    )
    dt = stepper.compute_stable_dt(state_n, cfl=0.05)

    stepper.step(
        state_n=state_n,
        state_np1=state_np1,
        dt=dt,
        left_boundary_state=left_boundary,
        right_boundary_state=right_boundary,
    )

    assert np.all(state_np1.A > 0.0)
    assert np.all(np.isfinite(state_np1.A))
    assert np.all(np.isfinite(state_np1.Q))


def test_step_rejects_negative_dt(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state_n = discretization.create_state(name="n")
    state_np1 = discretization.create_state(name="np1")
    discretization.interpolate_rest_state(state_n, physics)

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(discretization, physics)

    with pytest.raises(ValueError, match="dt must be positive"):
        stepper.step(
            state_n=state_n,
            state_np1=state_np1,
            dt=-1.0,
            left_boundary_state=boundary,
            right_boundary_state=boundary,
        )


def test_step_rejects_non_positive_area(
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
) -> None:
    state_n = discretization.create_state(name="n")
    state_np1 = discretization.create_state(name="np1")
    discretization.interpolate_rest_state(state_n, physics)

    state_n.A[3, 0] = 0.0

    boundary = BoundaryState(
        area=physics.params.area0,
        flow_rate=0.0,
    )

    stepper = DGLaxFriedrichsStepper(discretization, physics)

    with pytest.raises(RuntimeError, match="Non-positive area"):
        stepper.step(
            state_n=state_n,
            state_np1=state_np1,
            dt=1.0e-6,
            left_boundary_state=boundary,
            right_boundary_state=boundary,
        )


def _compute_residual_einsum_reference(
    *,
    physics: Hemo1DPhysics,
    stepper: DGLaxFriedrichsStepper,
    state,
    left_boundary_state: BoundaryState,
    right_boundary_state: BoundaryState,
) -> tuple[np.ndarray, np.ndarray]:
    interface_fluxes = compute_interface_fluxes(
        physics=physics,
        basis_left=stepper._basis_left,
        basis_right=stepper._basis_right,
        state=state,
        left_boundary_state=left_boundary_state,
        right_boundary_state=right_boundary_state,
    )

    area_q = state.A @ stepper._basis_quad.T
    flow_rate_q = state.Q @ stepper._basis_quad.T

    flux_q = physics.flux(area_q, flow_rate_q)
    source_q = physics.source(area_q, flow_rate_q)

    source_A = stepper._h_half * np.einsum(
        "q,cq,qm->cm",
        stepper._quad_weights,
        source_q[0],
        stepper._basis_quad,
        optimize=True,
    )
    source_Q = stepper._h_half * np.einsum(
        "q,cq,qm->cm",
        stepper._quad_weights,
        source_q[1],
        stepper._basis_quad,
        optimize=True,
    )

    flux_A = np.einsum(
        "q,cq,qm->cm",
        stepper._quad_weights,
        flux_q[0],
        stepper._dbasis_quad,
        optimize=True,
    )
    flux_Q = np.einsum(
        "q,cq,qm->cm",
        stepper._quad_weights,
        flux_q[1],
        stepper._dbasis_quad,
        optimize=True,
    )

    residual_A = -source_A + flux_A
    residual_Q = -source_Q + flux_Q

    residual_A += (
        interface_fluxes[:-1, 0][:, None] * stepper._basis_left[None, :]
        - interface_fluxes[1:, 0][:, None] * stepper._basis_right[None, :]
    )
    residual_Q += (
        interface_fluxes[:-1, 1][:, None] * stepper._basis_left[None, :]
        - interface_fluxes[1:, 1][:, None] * stepper._basis_right[None, :]
    )

    return residual_A, residual_Q
