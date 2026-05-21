from __future__ import annotations

import numpy as np
from mpi4py import MPI

from hemo1d.boundary import BoundaryCondition, CopyBoundaryCondition
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointSide
from hemo1d.solvers.base import VesselStepper
from hemo1d.solvers.cg.discretization import CGFEMDiscretization
from hemo1d.solvers.cg.forms import CGTaylorGalerkinFormBuilder
from hemo1d.solvers.cg.mass_solver import CGScalarMassSolver
from hemo1d.solvers.cg.state import CGState


class CGTaylorGalerkinStepper(VesselStepper):
    """
    One-step Taylor-Galerkin time stepper for the CG discretization.

    Responsibilities:
        1. compute stable dt estimate;
        2. compute endpoint boundary states;
        3. build Taylor-Galerkin RHS forms;
        4. solve scalar mass systems for A^{n+1} and Q^{n+1}.

    This class still works for a single vessel only.
    """

    def __init__(
        self,
        discretization: CGFEMDiscretization,
        physics: Hemo1DPhysics,
    ) -> None:
        self.discretization = discretization
        self.physics = physics

        self.form_builder = CGTaylorGalerkinFormBuilder(discretization, physics)
        self.mass_solver = CGScalarMassSolver(discretization)

    def compute_stable_dt(
        self,
        state: CGState,
        cfl: float = np.sqrt(3.0) / 3.0,
    ) -> float:
        """
        Compute the Taylor-Galerkin CFL estimate:

            dt <= sqrt(3)/3 * h / max(c_alpha + alpha |u|)

        where:
            u = Q/A.

        This matches the CFL structure used for the second-order
        Taylor-Galerkin scheme.
        """
        if cfl <= 0.0:
            raise ValueError("cfl must be positive.")

        A = state.A.x.array
        Q = state.Q.x.array

        if np.any(A <= 0.0):
            raise RuntimeError("Non-positive area encountered in CFL computation.")

        u_abs = np.abs(Q / A)
        speed = self.physics.c_alpha(A, Q) + self.physics.params.alpha * u_abs

        local_max = float(np.max(speed))
        global_max = self.discretization.comm.allreduce(local_max, op=MPI.MAX)

        if global_max <= 0.0:
            raise RuntimeError("Non-positive characteristic speed in CFL computation.")

        return cfl * self.discretization.h_min() / global_max

    def compute_boundary_states(
        self,
        state_n: CGState,
        t_np1: float,
        dt: float,
        left_boundary: BoundaryCondition,
        right_boundary: BoundaryCondition,
    ) -> tuple[BoundaryState, BoundaryState]:
        """
        Compute left and right BoundaryState objects using the boundary layer.
        """
        left_data = self.discretization.endpoint_data(state_n, EndpointSide.LEFT)
        right_data = self.discretization.endpoint_data(state_n, EndpointSide.RIGHT)

        left_state = left_boundary.compute(
            physics=self.physics,
            endpoint_data=left_data,
            side=EndpointSide.LEFT,
            t=t_np1,
            dt=dt,
        )
        right_state = right_boundary.compute(
            physics=self.physics,
            endpoint_data=right_data,
            side=EndpointSide.RIGHT,
            t=t_np1,
            dt=dt,
        )

        return left_state, right_state

    def step(
        self,
        state_n: CGState,
        state_np1: CGState,
        dt: float,
        left_boundary_state: BoundaryState,
        right_boundary_state: BoundaryState,
    ) -> None:
        """
        Advance one step using already-computed endpoint states.

        This method matches the generic VesselStepper interface.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        forms = self.form_builder.build(state_n=state_n, dt=dt)

        self.mass_solver.solve(
            rhs_form=forms.rhs_A,
            out=state_np1.A,
            left_value=left_boundary_state.area,
            right_value=right_boundary_state.area,
        )

        self.mass_solver.solve(
            rhs_form=forms.rhs_Q,
            out=state_np1.Q,
            left_value=left_boundary_state.flow_rate,
            right_value=right_boundary_state.flow_rate,
        )

        state_np1.scatter_forward()

    def step_with_boundary_conditions(
        self,
        state_n: CGState,
        state_np1: CGState,
        t_n: float,
        dt: float,
        left_boundary: BoundaryCondition | None = None,
        right_boundary: BoundaryCondition | None = None,
    ) -> None:
        """
        Convenience method for single-vessel runs.

        It computes boundary states and then advances the solution.

        Later, the network solver will compute boundary/junction states itself
        and call `step(...)` directly.
        """
        if left_boundary is None:
            left_boundary = CopyBoundaryCondition()
        if right_boundary is None:
            right_boundary = CopyBoundaryCondition()

        t_np1 = t_n + dt

        left_state, right_state = self.compute_boundary_states(
            state_n=state_n,
            t_np1=t_np1,
            dt=dt,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
        )

        self.step(
            state_n=state_n,
            state_np1=state_np1,
            dt=dt,
            left_boundary_state=left_state,
            right_boundary_state=right_state,
        )