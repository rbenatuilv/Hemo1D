from __future__ import annotations

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState
from hemo1d.solvers.dg.state import DGState


def compute_residual(
    physics: Hemo1DPhysics,
    basis_left: np.ndarray,
    basis_right: np.ndarray,
    basis_quad: np.ndarray,
    dbasis_quad: np.ndarray,
    quad_weights: np.ndarray,
    h_half: float,
    state: DGState,
    left_boundary_state: BoundaryState,
    right_boundary_state: BoundaryState,
    weighted_basis_quad: np.ndarray | None = None,
    weighted_dbasis_quad: np.ndarray | None = None,
    residual_A: np.ndarray | None = None,
    residual_Q: np.ndarray | None = None,
    interface_fluxes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the raw DG residual H(U_h), before applying M^{-1}.

    Returns:
        residual_A, residual_Q

    Both arrays have shape:
        (num_cells, num_local_dofs)
    """
    num_cells = state.num_cells
    num_local_dofs = state.num_local_dofs

    if residual_A is None:
        residual_A = np.empty((num_cells, num_local_dofs), dtype=np.float64)
    elif residual_A.shape != (num_cells, num_local_dofs):
        raise ValueError(
            f"residual_A has shape {residual_A.shape}, "
            f"expected {(num_cells, num_local_dofs)}."
        )

    if residual_Q is None:
        residual_Q = np.empty((num_cells, num_local_dofs), dtype=np.float64)
    elif residual_Q.shape != (num_cells, num_local_dofs):
        raise ValueError(
            f"residual_Q has shape {residual_Q.shape}, "
            f"expected {(num_cells, num_local_dofs)}."
        )

    if weighted_basis_quad is None:
        weighted_basis_quad = quad_weights[:, None] * basis_quad
    if weighted_dbasis_quad is None:
        weighted_dbasis_quad = quad_weights[:, None] * dbasis_quad

    interface_fluxes = compute_interface_fluxes(
        physics=physics,
        basis_left=basis_left,
        basis_right=basis_right,
        state=state,
        left_boundary_state=left_boundary_state,
        right_boundary_state=right_boundary_state,
        out=interface_fluxes,
    )

    area_q = state.A @ basis_quad.T
    flow_rate_q = state.Q @ basis_quad.T

    flux_q = physics.flux(area_q, flow_rate_q)
    source_q = physics.source(area_q, flow_rate_q)

    residual_A[:, :] = flux_q[0] @ weighted_dbasis_quad
    residual_A[:, :] -= h_half * (source_q[0] @ weighted_basis_quad)

    residual_Q[:, :] = flux_q[1] @ weighted_dbasis_quad
    residual_Q[:, :] -= h_half * (source_q[1] @ weighted_basis_quad)

    residual_A[:, :] += (
        interface_fluxes[:-1, 0][:, None] * basis_left[None, :]
        - interface_fluxes[1:, 0][:, None] * basis_right[None, :]
    )
    residual_Q[:, :] += (
        interface_fluxes[:-1, 1][:, None] * basis_left[None, :]
        - interface_fluxes[1:, 1][:, None] * basis_right[None, :]
    )

    return residual_A, residual_Q


def compute_interface_fluxes(
    physics: Hemo1DPhysics,
    basis_left: np.ndarray,
    basis_right: np.ndarray,
    state: DGState,
    left_boundary_state: BoundaryState,
    right_boundary_state: BoundaryState,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute numerical flux at every interface.

    fluxes[i] is oriented in the positive z direction.
    """
    left_trace_A = state.A @ basis_left
    left_trace_Q = state.Q @ basis_left
    right_trace_A = state.A @ basis_right
    right_trace_Q = state.Q @ basis_right

    left_A = np.empty(state.num_cells + 1, dtype=np.float64)
    left_Q = np.empty(state.num_cells + 1, dtype=np.float64)
    right_A = np.empty(state.num_cells + 1, dtype=np.float64)
    right_Q = np.empty(state.num_cells + 1, dtype=np.float64)

    left_A[0] = float(left_boundary_state.area)
    left_Q[0] = float(left_boundary_state.flow_rate)
    right_A[0] = left_trace_A[0]
    right_Q[0] = left_trace_Q[0]

    left_A[1:-1] = right_trace_A[:-1]
    left_Q[1:-1] = right_trace_Q[:-1]
    right_A[1:-1] = left_trace_A[1:]
    right_Q[1:-1] = left_trace_Q[1:]

    left_A[-1] = right_trace_A[-1]
    left_Q[-1] = right_trace_Q[-1]
    right_A[-1] = float(right_boundary_state.area)
    right_Q[-1] = float(right_boundary_state.flow_rate)

    flux_left = physics.flux(left_A, left_Q)
    flux_right = physics.flux(right_A, right_Q)

    lambda_plus_left, lambda_minus_left = physics.eigenvalues(left_A, left_Q)
    lambda_plus_right, lambda_minus_right = physics.eigenvalues(right_A, right_Q)

    speed_left = np.maximum(np.abs(lambda_plus_left), np.abs(lambda_minus_left))
    speed_right = np.maximum(np.abs(lambda_plus_right), np.abs(lambda_minus_right))
    speed_max = np.maximum(speed_left, speed_right)

    if out is None:
        fluxes = np.empty((state.num_cells + 1, 2), dtype=np.float64)
    elif out.shape != (state.num_cells + 1, 2):
        raise ValueError(
            f"out has shape {out.shape}, expected {(state.num_cells + 1, 2)}."
        )
    else:
        fluxes = out

    fluxes[:, 0] = (
        0.5 * (flux_left[0] + flux_right[0])
        - 0.5 * speed_max * (right_A - left_A)
    )
    fluxes[:, 1] = (
        0.5 * (flux_left[1] + flux_right[1])
        - 0.5 * speed_max * (right_Q - left_Q)
    )

    return fluxes


def max_speed_in_state(
    physics: Hemo1DPhysics,
    basis_left: np.ndarray,
    basis_right: np.ndarray,
    basis_quad: np.ndarray,
    state: DGState,
) -> float:
    """
    Return the maximum absolute characteristic speed in quadrature and trace points.
    """
    area_q = state.A @ basis_quad.T
    flow_rate_q = state.Q @ basis_quad.T

    lambda_plus_q, lambda_minus_q = physics.eigenvalues(area_q, flow_rate_q)
    speed_q = np.maximum(np.abs(lambda_plus_q), np.abs(lambda_minus_q))

    left_trace_A = state.A @ basis_left
    left_trace_Q = state.Q @ basis_left
    right_trace_A = state.A @ basis_right
    right_trace_Q = state.Q @ basis_right

    lambda_plus_l, lambda_minus_l = physics.eigenvalues(left_trace_A, left_trace_Q)
    lambda_plus_r, lambda_minus_r = physics.eigenvalues(right_trace_A, right_trace_Q)

    speed_l = np.maximum(np.abs(lambda_plus_l), np.abs(lambda_minus_l))
    speed_r = np.maximum(np.abs(lambda_plus_r), np.abs(lambda_minus_r))

    return float(max(np.max(speed_q), np.max(speed_l), np.max(speed_r)))
