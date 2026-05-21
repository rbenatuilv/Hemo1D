from __future__ import annotations

from typing import Any

from .backend import Backend
from .parameters import ModelParameters


class Hemo1DPhysics:
    """
    Pointwise physics for the 1D hemodynamics model in one compliant vessel.

    Unknowns:

        A(t, z): cross-sectional area
        Q(t, z): volumetric flow rate

    The pressure is eliminated through the elastic tube law:

        P - p_ext - p0 = psi(A; A0, beta)

    with:

        psi(A; A0, beta) = beta * (sqrt(A) - sqrt(A0)) / A0

    This class does not know anything about FEniCSx meshes, functions, or time
    stepping. It only provides the algebraic expressions needed by the solver.

    Because every operation goes through `self.math`, the same class works with:
        - NumPy, for unit tests and scalar/array checks;
        - UFL, for FEniCSx variational forms.
    """

    def __init__(self, model_params: ModelParameters, math_backend: Backend):
        self.params = model_params
        self.math = math_backend

    def psi(self, area: Any) -> Any:
        """
        Elastic part of the pressure-area law.

        Thesis notation:

            psi(A; A0, beta) = beta * (sqrt(A) - sqrt(A0)) / A0

        Therefore:

            psi(A0) = 0.
        """
        A0 = self.params.area0
        beta = self.params.beta

        return beta * (self.math.sqrt(area) - self.math.sqrt(A0)) / A0

    def pressure(self, area: Any) -> Any:
        """
        Mean internal pressure.

            P(A) = p_ext + p0 + psi(A)

        In many numerical tests, p0 = p_ext = 0, so pressure equals psi.
        """
        return self.params.p_ext + self.params.p0 + self.psi(area)

    def dpsi_dA(self, area: Any) -> Any:
        """
        Derivative of the tube law with respect to area.

        Since:

            psi(A) = beta * (sqrt(A) - sqrt(A0)) / A0

        then:

            dpsi/dA = beta / (2 A0 sqrt(A)).

        This is positive for A > 0 and beta > 0.
        """
        A0 = self.params.area0
        beta = self.params.beta

        return beta / (2.0 * A0 * self.math.sqrt(area))

    def velocity(self, area: Any, flow_rate: Any) -> Any:
        """
        Mean axial velocity.

            u = Q / A
        """
        return flow_rate / area

    def wave_speed(self, area: Any) -> Any:
        """
        Pulse wave speed.

        The 1D model defines:

            c(A) = sqrt( A / rho * dpsi/dA )

        For the selected tube law:

            c(A) = sqrt( beta * sqrt(A) / (2 rho A0) ).
        """
        rho = self.params.rho
        return self.math.sqrt(area * self.dpsi_dA(area) / rho)

    def c_alpha(self, area: Any, flow_rate: Any) -> Any:
        """
        Modified wave speed appearing in the eigenvalues.

            c_alpha = sqrt(c^2 + alpha(alpha - 1)u^2)

        where:
            u = Q/A.
        """
        alpha = self.params.alpha
        u = self.velocity(area, flow_rate)
        c = self.wave_speed(area)

        return self.math.sqrt(c * c + alpha * (alpha - 1.0) * u * u)

    def eigenvalues(self, area: Any, flow_rate: Any) -> tuple[Any, Any]:
        """
        Eigenvalues of the quasi-linear matrix H(U).

            lambda_plus  = alpha*u + c_alpha
            lambda_minus = alpha*u - c_alpha

        In the physiological subcritical regime, usually:

            lambda_plus  > 0
            lambda_minus < 0

        so one characteristic enters the vessel at each end.
        """
        alpha = self.params.alpha
        u = self.velocity(area, flow_rate)
        ca = self.c_alpha(area, flow_rate)

        return alpha * u + ca, alpha * u - ca

    def left_eigenvectors(
        self,
        area: Any,
        flow_rate: Any,
    ) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        """
        Unscaled left eigenvectors of H(U).

        We use:

            l_plus  = [ c_alpha - alpha*u, 1 ]
            l_minus = [ -c_alpha - alpha*u, 1 ]

        These will be useful for compatibility and non-reflecting boundary
        conditions.
        """
        alpha = self.params.alpha
        u = self.velocity(area, flow_rate)
        ca = self.c_alpha(area, flow_rate)

        l_plus = (ca - alpha * u, 1.0)
        l_minus = (-ca - alpha * u, 1.0)

        return l_plus, l_minus

    def friction_coefficient(self) -> float:
        """
        Friction coefficient K_r.

        For the velocity-profile family used in the thesis:

            K_r = 2 (gamma + 2) pi mu / rho

        For gamma = 2:

            K_r = 8 pi mu / rho.
        """
        gamma = self.params.gamma
        mu = self.params.mu
        rho = self.params.rho

        return 2.0 * (gamma + 2.0) * self.math.pi * mu / rho

    def C1(self, area: Any) -> Any:
        """
        Pressure primitive used in the conservative flux.

        The conservative formulation uses:

            F(U) = [ Q,
                     alpha Q^2/A + C1(A) ]^T

        where:

            C1(A) = integral from A0 to A of c(tau)^2 d tau.

        Since:

            c(A)^2 = beta * sqrt(A) / (2 rho A0),

        we get:

            C1(A) = beta / (3 rho A0) * (A^(3/2) - A0^(3/2)).

        Therefore:

            C1(A0) = 0.
        """
        A0 = self.params.area0
        beta = self.params.beta
        rho = self.params.rho

        return beta * (
            area * self.math.sqrt(area) - A0 * self.math.sqrt(A0)
        ) / (3.0 * rho * A0)

    def flux(self, area: Any, flow_rate: Any) -> Any:
        """
        Conservative flux F(U), with U = [A, Q]^T.

        The conservative form is:

            dU/dt + dF(U)/dz + S(U) = 0

        with:

            F_1 = Q
            F_2 = alpha Q^2/A + C1(A)

        Important:
            The second component is not Q*u + P. The pressure term enters
            through C1(A), the primitive of c(A)^2.
        """
        alpha = self.params.alpha

        F1 = flow_rate
        F2 = alpha * flow_rate * flow_rate / area + self.C1(area)

        return self.math.vector([F1, F2])

    def source(self, area: Any, flow_rate: Any) -> Any:
        """
        Source term S(U) for a vessel with constant A0 and beta.

        For now we assume A0 and beta are constant along the vessel, so the only
        source is friction:

            S_1 = 0
            S_2 = K_r Q/A

        Later, if we allow tapering or beta(z), this method must be extended with
        the geometric/material source terms.
        """
        Kr = self.friction_coefficient()

        S1 = 0.0 * area
        S2 = Kr * flow_rate / area

        return self.math.vector([S1, S2])

    def H_matrix(self, area: Any, flow_rate: Any) -> Any:
        """
        Quasi-linear matrix H(U).

        The quasi-linear form is:

            dU/dt + H(U) dU/dz + S(U) = 0

        where:

            H = [[0, 1],
                 [c^2 - alpha*(Q/A)^2, 2 alpha Q/A]]
        """
        alpha = self.params.alpha

        u = self.velocity(area, flow_rate)
        c = self.wave_speed(area)

        H11 = 0.0
        H12 = 1.0
        H21 = c * c - alpha * u * u
        H22 = 2.0 * alpha * u

        return self.math.matrix([
            [H11, H12],
            [H21, H22],
        ])

    def total_pressure(
        self,
        area: Any,
        flow_rate: Any,
        include_density: bool = True,
    ) -> Any:
        """
        Total pressure used later for interfaces and branching.

        Dimensionally consistent form:

            P_tot = P + 1/2 rho u^2

        Some references omit rho depending on convention or nondimensionalization.
        By default we keep rho.
        """
        u = self.velocity(area, flow_rate)

        dynamic = 0.5 * u * u
        if include_density:
            dynamic = self.params.rho * dynamic

        return self.pressure(area) + dynamic