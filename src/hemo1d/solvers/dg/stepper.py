from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState
from hemo1d.solvers.base import VesselStepper
from hemo1d.solvers.dg.discretization import DGFEMDiscretization
from hemo1d.solvers.dg.flux import DGFluxScheme, canonicalize_dg_flux_scheme
from hemo1d.solvers.dg.limiter import (
    DGLimiterConfig,
    DGLimiterStats,
    DGSlopeLimiter,
)
from hemo1d.solvers.dg.residual import (
    compute_residual,
    max_speed_in_state,
)
from hemo1d.solvers.dg.state import DGState


TimeScheme = Literal["euler", "rk2"]


@dataclass(frozen=True)
class DGRHS:
    """
    Element-local DG RHS coefficients.

    These arrays represent M^{-1} H(U_h), where H is the raw DG residual.

    Shapes:
        dA_dt: (num_cells, num_local_dofs)
        dQ_dt: (num_cells, num_local_dofs)
    """

    dA_dt: np.ndarray
    dQ_dt: np.ndarray


@dataclass(frozen=True)
class DGStepStats:
    """
    Diagnostics for the last DG step.
    """

    stage_limiter: DGLimiterStats | None = None
    final_limiter: DGLimiterStats | None = None


class DGLaxFriedrichsStepper(VesselStepper):
    """
    Explicit DG stepper with selectable DG interface flux.

    Spatial discretization:
        polynomial DG on each cell, degree 0 or degree 1.

    Numerical flux:
        - "lxf": local Lax-Friedrichs / Rusanov
        - "hll": HLL

    Time discretization:
        - "euler": forward Euler
        - "rk2": two-stage explicit RK2

    Stabilization:
        For DG1, a slope/positivity limiter is applied after each explicit
        stage. This is essential: the explicit DG update can create negative
        endpoint values even if the cell average is still positive.

    PDE convention in this codebase:
        U_t + F(U)_z + S(U) = 0

    Therefore the weak DG residual is:

        ∫_Ie U_t · Phi
        =
        - ∫_Ie S(U_h) · Phi
        + ∫_Ie F(U_h) · Phi_x
        - F*_{e+1/2} · Phi(x^-_{e+1/2})
        + F*_{e-1/2} · Phi(x^+_{e-1/2})
    """

    def __init__(
        self,
        discretization: DGFEMDiscretization,
        physics: Hemo1DPhysics,
        time_scheme: TimeScheme = "rk2",
        limiter_config: DGLimiterConfig | None = None,
        flux_scheme: DGFluxScheme | str = "lxf",
    ) -> None:
        if time_scheme not in ("euler", "rk2"):
            raise ValueError("time_scheme must be either 'euler' or 'rk2'.")

        self.discretization = discretization
        self.physics = physics
        self.time_scheme = time_scheme
        self.flux_scheme = canonicalize_dg_flux_scheme(flux_scheme)

        self.limiter = DGSlopeLimiter(
            self._default_limiter_config(limiter_config)
        )

        # Cached views/constants used by vectorized kernels.
        self._basis_left = self.discretization.basis_at_left
        self._basis_right = self.discretization.basis_at_right
        self._basis_quad = self.discretization.basis_at_quad
        self._dbasis_quad = self.discretization.basis_derivative_at_quad
        self._quad_weights = self.discretization.quad_weights
        self._h_half = 0.5 * self.discretization.h
        self._invM_T = self.discretization.inverse_mass_matrix.T
        self._weighted_basis_quad = self._quad_weights[:, None] * self._basis_quad
        self._weighted_dbasis_quad = self._quad_weights[:, None] * self._dbasis_quad

        # Work state used only by RK2.
        self._stage_state = discretization.create_state(name="dg_stage")

        # Reused work arrays for residual/RHS kernels.
        shape = (self.discretization.num_cells, self.discretization.num_local_dofs)
        self._residual_A = np.empty(shape, dtype=np.float64)
        self._residual_Q = np.empty(shape, dtype=np.float64)
        self._rhs_A = np.empty(shape, dtype=np.float64)
        self._rhs_Q = np.empty(shape, dtype=np.float64)
        self._interface_fluxes = np.empty(
            (self.discretization.num_cells + 1, 2),
            dtype=np.float64,
        )

        self.last_step_stats: DGStepStats | None = None

    def compute_stable_dt(
        self,
        state: DGState,
        cfl: float = 0.5,
    ) -> float:
        """
        Compute a practical explicit DG time-step estimate.

        For DG degree p:

            dt <= cfl * h / ((2p + 1) * max |lambda|)

        For DG1, the (2p + 1) factor is important.
        """
        if cfl <= 0.0:
            raise ValueError("cfl must be positive.")

        self._check_state_compatible(state)
        self._check_usable_state_for_flux(state)

        max_speed = max_speed_in_state(
            physics=self.physics,
            basis_left=self._basis_left,
            basis_right=self._basis_right,
            basis_quad=self._basis_quad,
            state=state,
        )
        if max_speed <= 0.0:
            raise RuntimeError("Non-positive characteristic speed in CFL computation.")

        degree_factor = 2 * self.discretization.degree + 1
        return cfl * self.discretization.h_min() / (degree_factor * max_speed)

    def step(
        self,
        state_n: DGState,
        state_np1: DGState,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        """
        Advance one time step using the configured explicit time scheme.
        """
        if self.time_scheme == "euler":
            self.step_euler(
                state_n=state_n,
                state_np1=state_np1,
                dt=dt,
                left_boundary_state=left_boundary_state,
                right_boundary_state=right_boundary_state,
            )
            return

        if self.time_scheme == "rk2":
            self.step_rk2(
                state_n=state_n,
                state_np1=state_np1,
                dt=dt,
                left_boundary_state=left_boundary_state,
                right_boundary_state=right_boundary_state,
            )
            return

        raise RuntimeError(f"Unexpected time scheme: {self.time_scheme}")

    def step_euler(
        self,
        state_n: DGState,
        state_np1: DGState,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        """
        Forward Euler update:

            U^{n+1} = U^n + dt * L(U^n)

        Then limit U^{n+1}.
        """
        self._check_step_inputs(
            state_n=state_n,
            state_np1=state_np1,
            dt=dt,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
        )

        self._compute_rhs_into(
            state=state_n,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            dA_dt=self._rhs_A,
            dQ_dt=self._rhs_Q,
            validate=False,
        )

        state_np1.A[:, :] = state_n.A + dt * self._rhs_A
        state_np1.Q[:, :] = state_n.Q + dt * self._rhs_Q

        final_stats = self._limit_and_validate_state(state_np1)

        self.last_step_stats = DGStepStats(
            stage_limiter=None,
            final_limiter=final_stats,
        )

    def step_rk2(
        self,
        state_n: DGState,
        state_np1: DGState,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        """
        Two-stage explicit RK2 update.

        Stage 1:
            V = U^n + dt * L(U^n)
            limit(V)

        Stage 2:
            U^{n+1} = 0.5 U^n + 0.5 V + 0.5 dt * L(V)
            limit(U^{n+1})

        The limiter is applied after both explicit stages. Without this, DG1 can
        generate invalid endpoint values before the second RHS evaluation.
        """
        self._check_step_inputs(
            state_n=state_n,
            state_np1=state_np1,
            dt=dt,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
        )

        self._compute_rhs_into(
            state=state_n,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            dA_dt=self._rhs_A,
            dQ_dt=self._rhs_Q,
            validate=False,
        )

        stage = self._stage_state
        stage.A[:, :] = state_n.A + dt * self._rhs_A
        stage.Q[:, :] = state_n.Q + dt * self._rhs_Q

        stage_stats = self._limit_and_validate_state(stage)

        # For now we use the same exterior states at the RK2 stage.
        # Later, if needed, the NetworkSolver can compute stage-time boundary
        # states and pass them into a dedicated stage API.
        self._compute_rhs_into(
            state=stage,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            dA_dt=self._rhs_A,
            dQ_dt=self._rhs_Q,
            validate=False,
        )

        state_np1.A[:, :] = (
            0.5 * state_n.A
            + 0.5 * stage.A
            + 0.5 * dt * self._rhs_A
        )
        state_np1.Q[:, :] = (
            0.5 * state_n.Q
            + 0.5 * stage.Q
            + 0.5 * dt * self._rhs_Q
        )

        final_stats = self._limit_and_validate_state(state_np1)

        self.last_step_stats = DGStepStats(
            stage_limiter=stage_stats,
            final_limiter=final_stats,
        )

    def compute_rhs(
        self,
        state: DGState,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> DGRHS:
        """
        Compute M^{-1} H(U_h), where H is the raw DG residual.

        Returns coefficient time derivatives dA/dt and dQ/dt.
        """
        dA_dt = np.empty_like(state.A)
        dQ_dt = np.empty_like(state.Q)
        self._compute_rhs_into(
            state=state,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            dA_dt=dA_dt,
            dQ_dt=dQ_dt,
        )

        return DGRHS(
            dA_dt=dA_dt,
            dQ_dt=dQ_dt,
        )

    def _compute_rhs_into(
        self,
        state: DGState,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
        dA_dt: np.ndarray,
        dQ_dt: np.ndarray,
        validate: bool = True,
    ) -> None:
        if validate:
            self._check_state_compatible(state)
            self._check_usable_state_for_flux(state)
            self._check_boundary_state(left_boundary_state, name="left_boundary_state")
            self._check_boundary_state(right_boundary_state, name="right_boundary_state")

        residual_A, residual_Q = compute_residual(
            physics=self.physics,
            basis_left=self._basis_left,
            basis_right=self._basis_right,
            basis_quad=self._basis_quad,
            dbasis_quad=self._dbasis_quad,
            quad_weights=self._quad_weights,
            h_half=self._h_half,
            state=state,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            weighted_basis_quad=self._weighted_basis_quad,
            weighted_dbasis_quad=self._weighted_dbasis_quad,
            residual_A=self._residual_A,
            residual_Q=self._residual_Q,
            interface_fluxes=self._interface_fluxes,
            flux_scheme=self.flux_scheme,
        )

        np.matmul(residual_A, self._invM_T, out=dA_dt)
        np.matmul(residual_Q, self._invM_T, out=dQ_dt)

    def compute_residual(
        self,
        state: DGState,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the raw DG residual H(U_h), before applying M^{-1}.

        Returns:
            residual_A, residual_Q

        Both arrays have shape:
            (num_cells, num_local_dofs)
        """
        self._check_state_compatible(state)
        self._check_usable_state_for_flux(state)

        return compute_residual(
            physics=self.physics,
            basis_left=self._basis_left,
            basis_right=self._basis_right,
            basis_quad=self._basis_quad,
            dbasis_quad=self._dbasis_quad,
            quad_weights=self._quad_weights,
            h_half=self._h_half,
            state=state,
            left_boundary_state=left_boundary_state,
            right_boundary_state=right_boundary_state,
            weighted_basis_quad=self._weighted_basis_quad,
            weighted_dbasis_quad=self._weighted_dbasis_quad,
            flux_scheme=self.flux_scheme,
        )

    def _limit_and_validate_state(self, state: DGState) -> DGLimiterStats:
        """
        Apply limiter in-place and verify the resulting state can be used in fluxes.
        """
        stats = self.limiter.apply(state)
        self._check_usable_state_for_flux(state)
        state.scatter_forward()
        return stats

    def _default_limiter_config(
        self,
        config: DGLimiterConfig | None,
    ) -> DGLimiterConfig:
        """
        Build a limiter config.

        If no config is provided, choose a tiny area floor relative to A0.
        This avoids hard-coding an absolute physiological scale.
        """
        if config is not None:
            return config

        area0 = float(self.physics.params.area0)
        area_floor = max(1.0e-14, 1.0e-10 * area0)

        return DGLimiterConfig(
            enabled=True,
            slope=True,
            positivity=True,
            area_floor=area_floor,
            minmod_beta=1.0,
            limit_area=True,
            limit_flow_rate=True,
            raise_on_bad_average=True,
        )

    def _check_step_inputs(
        self,
        state_n: DGState,
        state_np1: DGState,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self._check_state_compatible(state_n)
        self._check_state_compatible(state_np1)

        self._check_usable_state_for_flux(state_n)

        self._check_boundary_state(left_boundary_state, name="left_boundary_state")
        self._check_boundary_state(right_boundary_state, name="right_boundary_state")

    def _check_state_compatible(self, state: DGState) -> None:
        if state.num_cells != self.discretization.num_cells:
            raise ValueError(
                f"State has {state.num_cells} cells, but discretization has "
                f"{self.discretization.num_cells} cells."
            )
        if state.degree != self.discretization.degree:
            raise ValueError(
                f"State has degree {state.degree}, but discretization has "
                f"degree {self.discretization.degree}."
            )

    def _check_usable_state_for_flux(self, state: DGState) -> None:
        """
        Check that all nodal values are finite and A is above the limiter floor.

        The physics flux and eigenvalues need positive area everywhere they are
        evaluated. For DG1 this includes traces and quadrature points. Since the
        quadrature values are convex combinations of nodal values for degree 1,
        positive nodal values are enough here.
        """
        state.assert_positive_area(self.limiter.config.area_floor)

    def _check_boundary_state(self, boundary_state: BoundaryState, name: str) -> None:
        if not np.isfinite(boundary_state.area):
            raise RuntimeError(f"{name}.area is not finite.")
        if not np.isfinite(boundary_state.flow_rate):
            raise RuntimeError(f"{name}.flow_rate is not finite.")
        if boundary_state.area <= self.limiter.config.area_floor:
            raise RuntimeError(
                f"{name}.area is too small/non-positive: "
                f"A={boundary_state.area:.16e}, "
                f"floor={self.limiter.config.area_floor:.16e}."
            )
