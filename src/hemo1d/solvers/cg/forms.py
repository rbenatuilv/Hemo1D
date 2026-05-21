from __future__ import annotations

from dataclasses import dataclass

import ufl
from dolfinx import fem

from hemo1d.core.backend import UFL_BACKEND
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.cg.discretization import CGFEMDiscretization
from hemo1d.solvers.cg.state import CGState


@dataclass(frozen=True)
class CGTaylorGalerkinForms:
    """
    UFL forms for one explicit Taylor-Galerkin step.

    The forms are scalar because A and Q are stored as separate scalar CG
    functions.

    mass:
        Scalar mass form: (u, v)

    rhs_A:
        Right-hand side for the area equation.

    rhs_Q:
        Right-hand side for the flow-rate equation.
    """

    mass: fem.Form
    rhs_A: fem.Form
    rhs_Q: fem.Form


class CGTaylorGalerkinFormBuilder:
    """
    Build the UFL forms for the CG Taylor-Galerkin method.

    This class only builds forms. It does not:
        - assemble matrices,
        - solve linear systems,
        - impose boundary conditions,
        - advance time.

    Those responsibilities will belong to later layers.
    """

    def __init__(
        self,
        discretization: CGFEMDiscretization,
        physics: Hemo1DPhysics,
    ) -> None:
        self.discretization = discretization

        # Same physical parameters, but UFL operations instead of NumPy.
        self.physics = Hemo1DPhysics(
            model_params=physics.params,
            math_backend=UFL_BACKEND,
        )

        # Create reusable Functions and a dt Constant so UFL forms can be
        # assembled once and reused each time by updating these objects' values.
        V = self.discretization.V
        domain = self.discretization.domain
        dx = ufl.dx(domain=domain)

        # Placeholder functions that will be updated in `build`.
        self._A_fun = fem.Function(V)
        self._Q_fun = fem.Function(V)

        # Forms will be built lazily on first call to build() and cached for
        # reuse as long as the time-step `dt` does not change.
        self._mass_form = None
        self._rhs_A_form = None
        self._rhs_Q_form = None
        self._built_dt = None

    def build(self, state_n: CGState, dt: float) -> CGTaylorGalerkinForms:
        """
        Build the Taylor-Galerkin forms evaluated explicitly at `state_n`.

        In this optimized implementation, symbolic forms are cached and reused.
        The only per-call updates are the values of A, Q and (if needed) dt.

        Parameters
        ----------
        state_n:
            State at time t^n.

        dt:
            Time step.

        Returns
        -------
        CGTaylorGalerkinForms
            Mass form and RHS forms for A and Q.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        # Copy current state values into the reusable Functions.
        self._A_fun.x.array[:] = state_n.A.x.array[:]
        self._A_fun.x.scatter_forward()

        self._Q_fun.x.array[:] = state_n.Q.x.array[:]
        self._Q_fun.x.scatter_forward()

        # If forms haven't been created yet for this dt, build them now and
        # cache them. For a fixed `dt` (the common case for the example), this
        # avoids rebuilding symbolic UFL expressions every time step.
        if self._mass_form is None or self._built_dt != float(dt):
            trial = ufl.TrialFunction(self.discretization.V)
            test = ufl.TestFunction(self.discretization.V)

            A = self._A_fun
            Q = self._Q_fun

            physics = self.physics

            F = physics.flux(A, Q)
            S = physics.source(A, Q)
            H = physics.H_matrix(A, Q)

            dF_dz = ufl.as_vector([F[0].dx(0), F[1].dx(0)])

            HS = ufl.dot(H, S)
            HdF = ufl.dot(H, dF_dz)

            Kr = physics.friction_coefficient()

            # For the current single-vessel model, A0 and beta are constant.
            # Therefore:
            #
            #     S(U) = [0, Kr Q/A]^T
            #
            # and:
            #
            #     dS/dU = [[0, 0],
            #              [-Kr Q/A^2, Kr/A]]
            #
            source_jacobian = ufl.as_matrix(
                [
                    [0.0, 0.0],
                    [-Kr * Q / (A * A), Kr / A],
                ]
            )

            dS_dU_dF = ufl.dot(source_jacobian, dF_dz)
            dS_dU_S = ufl.dot(source_jacobian, S)

            # Taylor-Galerkin corrected flux and source:
            #
            # FLW = F - dt/2 H S
            # SLW = S - dt/2 (dS/dU) S
            #
            FLW = F - 0.5 * float(dt) * HS
            SLW = S - 0.5 * float(dt) * dS_dU_S

            dx = ufl.dx(domain=self.discretization.domain)

            mass = fem.form(trial * test * dx)

            rhs_A = fem.form(
                A * test * dx
                + float(dt) * FLW[0] * test.dx(0) * dx
                + 0.5 * float(dt) * float(dt) * dS_dU_dF[0] * test * dx
                - 0.5 * float(dt) * float(dt) * HdF[0] * test.dx(0) * dx
                - float(dt) * SLW[0] * test * dx
            )

            rhs_Q = fem.form(
                Q * test * dx
                + float(dt) * FLW[1] * test.dx(0) * dx
                + 0.5 * float(dt) * float(dt) * dS_dU_dF[1] * test * dx
                - 0.5 * float(dt) * float(dt) * HdF[1] * test.dx(0) * dx
                - float(dt) * SLW[1] * test * dx
            )

            self._mass_form = mass
            self._rhs_A_form = rhs_A
            self._rhs_Q_form = rhs_Q
            self._built_dt = float(dt)

        return CGTaylorGalerkinForms(
            mass=self._mass_form,
            rhs_A=self._rhs_A_form,
            rhs_Q=self._rhs_Q_form,
        )