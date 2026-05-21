from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide
from hemo1d.core.newton import NewtonConfig, NewtonResult, NewtonSolver


@dataclass(frozen=True)
class JunctionEndpointData:
    """
    Data associated with one vessel endpoint connected to a junction.

    physics:
        Physical model of the corresponding vessel.

    endpoint_data:
        State and spatial derivative at the endpoint, evaluated at t^n.

    side:
        LEFT or RIGHT endpoint of the vessel.

    name:
        Human-readable label, useful for debugging.
    """

    physics: Hemo1DPhysics
    endpoint_data: EndpointData
    side: EndpointSide
    name: str
    angle: float | None = None


@dataclass(frozen=True)
class BifurcationJunctionData:
    """
    Data for a simple 1-to-2 bifurcation.

    Convention:
        parent.RIGHT -> junction
        daughter1.LEFT -> junction
        daughter2.LEFT -> junction

    The implementation itself uses the endpoint side, so the orientation is not
    hard-coded in the compatibility helper.
    """

    parent: JunctionEndpointData
    daughter1: JunctionEndpointData
    daughter2: JunctionEndpointData


@dataclass(frozen=True)
class BifurcationSolution:
    """
    Solved endpoint states at a bifurcation.
    """

    parent: BoundaryState
    daughter1: BoundaryState
    daughter2: BoundaryState
    newton_result: NewtonResult


def _state_vector(area: float, flow_rate: float) -> np.ndarray:
    return np.array([area, flow_rate], dtype=float)


def _mass_sign(side: EndpointSide) -> float:
    """
    Sign of Q in the junction mass balance.

    Q is positive in the local vessel coordinate direction z=0 -> z=L.

    At a junction:
        RIGHT endpoint contributes +Q
        LEFT endpoint contributes -Q

    Therefore mass conservation is:

        sum_i sign_i * Q_i = 0
    """
    if side == EndpointSide.RIGHT:
        return +1.0

    if side == EndpointSide.LEFT:
        return -1.0

    raise ValueError(f"Unknown endpoint side: {side}")


def compatibility_target(
    physics: Hemo1DPhysics,
    endpoint_data: EndpointData,
    dt: float,
) -> np.ndarray:
    """
    Explicit compatibility target:

        CC = U^n - dt * ( H(U^n) dU^n/dz + S(U^n) )

    The eigenvectors and CC are evaluated explicitly at t^n, while the junction
    unknowns A^{n+1}, Q^{n+1} appear in l^T U^{n+1}.
    """
    A = endpoint_data.state.area
    Q = endpoint_data.state.flow_rate

    U = _state_vector(A, Q)
    dU_dz = _state_vector(endpoint_data.d_area_dz, endpoint_data.d_flow_rate_dz)

    H = physics.H_matrix(A, Q)
    S = physics.source(A, Q)

    return U - dt * (H @ dU_dz + S)


def outgoing_left_eigenvector(
    physics: Hemo1DPhysics,
    endpoint_data: EndpointData,
    side: EndpointSide,
) -> np.ndarray:
    """
    Outgoing-characteristic left eigenvector at the endpoint.

    Under subcritical flow:
        lambda_plus  > 0
        lambda_minus < 0

    LEFT endpoint:
        outgoing from the vessel domain is lambda_minus.

    RIGHT endpoint:
        outgoing from the vessel domain is lambda_plus.
    """
    A = endpoint_data.state.area
    Q = endpoint_data.state.flow_rate

    l_plus, l_minus = physics.left_eigenvectors(A, Q)

    if side == EndpointSide.LEFT:
        return np.array(l_minus, dtype=float)

    if side == EndpointSide.RIGHT:
        return np.array(l_plus, dtype=float)

    raise ValueError(f"Unknown endpoint side: {side}")


def _total_pressure_gradient(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    include_density: bool,
) -> tuple[float, float]:
    """
    Gradient of total pressure with respect to (A, Q).

    This matches Hemo1DPhysics.total_pressure():

        Ptot = P(A) + 0.5 * rho * (Q/A)^2      if include_density=True
        Ptot = P(A) + 0.5       * (Q/A)^2      if include_density=False
    """
    if area <= 0.0:
        raise ValueError(
            f"Cannot evaluate total-pressure gradient with non-positive area: A={area}"
        )

    kinetic_factor = physics.params.rho if include_density else 1.0

    dP_dA = float(physics.dpsi_dA(area))

    dK_dA = -kinetic_factor * flow_rate**2 / area**3
    dK_dQ = kinetic_factor * flow_rate / area**2

    return dP_dA + dK_dA, dK_dQ


def _pressure_loss_term(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    angle: float | None,
) -> float:
    """
    Pressure loss term for a junction with angle/losses.
    """
    if angle is None:
        return 0.0

    gamma = physics.params.gamma_pressure
    if gamma == 0.0:
        return 0.0

    return gamma * flow_rate * abs(flow_rate) / area**2 * np.sqrt(2.0 * (1.0 - np.cos(angle)))


def _d_pressure_loss_term_gradient(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    angle: float | None,
) -> tuple[float, float]:
    """
    Gradient of the pressure loss term with respect to (A, Q).
    """
    if angle is None:
        return 0.0, 0.0

    gamma = physics.params.gamma_pressure
    if gamma == 0.0:
        return 0.0, 0.0

    common_factor = gamma * abs(flow_rate) / area**2 * np.sqrt(2.0 * (1.0 - np.cos(angle)))

    d_loss_dA = -2.0 * common_factor * flow_rate / area
    d_loss_dQ = 2.0 * common_factor

    return d_loss_dA, d_loss_dQ

class BifurcationJunctionResidual:
    """
    Residual of the 6-equation bifurcation system.

    Unknown vector:

        x = [A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2]

    Equations:

        R0 = Q_p - Q_d1 - Q_d2

        R1 = Ptot_p - Ptot_d1
        R2 = Ptot_p - Ptot_d2

        R3 = l_p^T U_p   - l_p^T CC_p
        R4 = l_d1^T U_d1 - l_d1^T CC_d1
        R5 = l_d2^T U_d2 - l_d2^T CC_d2

    This first version assumes total-pressure continuity, i.e. no angle/loss
    terms yet.
    """

    def __init__(
        self,
        data: BifurcationJunctionData,
        dt: float,
        include_density_in_total_pressure: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.data = data
        self.dt = dt
        self.include_density_in_total_pressure = include_density_in_total_pressure

        self._cc_parent = compatibility_target(
            data.parent.physics,
            data.parent.endpoint_data,
            dt,
        )
        self._cc_daughter1 = compatibility_target(
            data.daughter1.physics,
            data.daughter1.endpoint_data,
            dt,
        )
        self._cc_daughter2 = compatibility_target(
            data.daughter2.physics,
            data.daughter2.endpoint_data,
            dt,
        )

        self._l_parent = outgoing_left_eigenvector(
            data.parent.physics,
            data.parent.endpoint_data,
            data.parent.side,
        )
        self._l_daughter1 = outgoing_left_eigenvector(
            data.daughter1.physics,
            data.daughter1.endpoint_data,
            data.daughter1.side,
        )
        self._l_daughter2 = outgoing_left_eigenvector(
            data.daughter2.physics,
            data.daughter2.endpoint_data,
            data.daughter2.side,
        )

    def initial_guess(self) -> np.ndarray:
        """
        Previous endpoint values are the natural Newton initial guess.
        """
        p = self.data.parent.endpoint_data.state
        d1 = self.data.daughter1.endpoint_data.state
        d2 = self.data.daughter2.endpoint_data.state

        return np.array(
            [
                p.area,
                p.flow_rate,
                d1.area,
                d1.flow_rate,
                d2.area,
                d2.flow_rate,
            ],
            dtype=float,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if x.shape != (6,):
            raise ValueError("Bifurcation unknown vector must have shape (6,).")

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = x

        if A_p <= 0.0 or A_d1 <= 0.0 or A_d2 <= 0.0:
            # Newton may briefly try invalid states. Return a large residual
            # instead of evaluating sqrt on negative areas.
            return np.full(6, 1.0e20, dtype=float)

        physics_p = self.data.parent.physics
        physics_d1 = self.data.daughter1.physics
        physics_d2 = self.data.daughter2.physics

        ptot_p = physics_p.total_pressure(
            A_p,
            Q_p,
            include_density=self.include_density_in_total_pressure,
        )
        ptot_d1 = physics_d1.total_pressure(
            A_d1,
            Q_d1,
            include_density=self.include_density_in_total_pressure,
        )
        ptot_d2 = physics_d2.total_pressure(
            A_d2,
            Q_d2,
            include_density=self.include_density_in_total_pressure,
        )

        U_p = _state_vector(A_p, Q_p)
        U_d1 = _state_vector(A_d1, Q_d1)
        U_d2 = _state_vector(A_d2, Q_d2)

        residual = np.zeros(6, dtype=float)

        s1 = _mass_sign(self.data.parent.side)
        s2 = _mass_sign(self.data.daughter1.side)
        s3 = _mass_sign(self.data.daughter2.side)

        residual[0] = s1 * Q_p + s2 * Q_d1 + s3 * Q_d2
        residual[1] = ptot_p - ptot_d1 - _pressure_loss_term(
            physics_d1,
            A_d1,
            Q_d1,
            self.data.daughter1.angle,
        )
        residual[2] = ptot_p - ptot_d2 - _pressure_loss_term(
            physics_d2,
            A_d2,
            Q_d2,
            self.data.daughter2.angle,
        )

        residual[3] = float(self._l_parent @ U_p - self._l_parent @ self._cc_parent)
        residual[4] = float(self._l_daughter1 @ U_d1 - self._l_daughter1 @ self._cc_daughter1)
        residual[5] = float(self._l_daughter2 @ U_d2 - self._l_daughter2 @ self._cc_daughter2)

        return residual

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic Jacobian of the 6-equation bifurcation residual.

        Unknown vector:

            x = [A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2]
        """
        x = np.asarray(x, dtype=float)

        if x.shape != (6,):
            raise ValueError("Bifurcation unknown vector must have shape (6,).")

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = x

        if A_p <= 0.0 or A_d1 <= 0.0 or A_d2 <= 0.0:
            # Do not build a fake Jacobian for an invalid physical state.
            # Let Newton fail clearly instead of producing misleading steps.
            raise ValueError(
                "Cannot evaluate bifurcation Jacobian with non-positive area: "
                f"A_p={A_p}, A_d1={A_d1}, A_d2={A_d2}"
            )

        physics_p = self.data.parent.physics
        physics_d1 = self.data.daughter1.physics
        physics_d2 = self.data.daughter2.physics

        dptot_p_dA, dptot_p_dQ = _total_pressure_gradient(
            physics_p,
            A_p,
            Q_p,
            include_density=self.include_density_in_total_pressure,
        )
        dptot_d1_dA, dptot_d1_dQ = _total_pressure_gradient(
            physics_d1,
            A_d1,
            Q_d1,
            include_density=self.include_density_in_total_pressure,
        )
        dptot_d2_dA, dptot_d2_dQ = _total_pressure_gradient(
            physics_d2,
            A_d2,
            Q_d2,
            include_density=self.include_density_in_total_pressure,
        )

        J = np.zeros((6, 6), dtype=float)

        s1 = _mass_sign(self.data.parent.side)
        s2 = _mass_sign(self.data.daughter1.side)
        s3 = _mass_sign(self.data.daughter2.side)

        dF2_d1 = _d_pressure_loss_term_gradient(
            physics_d1,
            A_d1,
            Q_d1,
            self.data.daughter1.angle,
        )
        dF_d2 = _d_pressure_loss_term_gradient(
            physics_d2,
            A_d2,
            Q_d2,
            self.data.daughter2.angle,
        )

        # R0 = s1 Q_p + s2 Q_d1 + s3 Q_d2
        J[0, 1] = s1
        J[0, 3] = s2
        J[0, 5] = s3

        # R1 = Ptot_p - Ptot_d1
        J[1, 0] = dptot_p_dA
        J[1, 1] = dptot_p_dQ
        J[1, 2] = -dptot_d1_dA - dF2_d1[0]
        J[1, 3] = -dptot_d1_dQ - dF2_d1[1]

        # R2 = Ptot_p - Ptot_d2
        J[2, 0] = dptot_p_dA
        J[2, 1] = dptot_p_dQ
        J[2, 4] = -dptot_d2_dA - dF_d2[0]
        J[2, 5] = -dptot_d2_dQ - dF_d2[1]

        # R3 = l_p^T U_p - const
        J[3, 0] = self._l_parent[0]
        J[3, 1] = self._l_parent[1]

        # R4 = l_d1^T U_d1 - const
        J[4, 2] = self._l_daughter1[0]
        J[4, 3] = self._l_daughter1[1]

        # R5 = l_d2^T U_d2 - const
        J[5, 4] = self._l_daughter2[0]
        J[5, 5] = self._l_daughter2[1]

        return J
    

class BifurcationJunctionSolver:
    """
    Newton-based solver for one 1-to-2 bifurcation.
    """

    def __init__(
        self,
        newton_solver: NewtonSolver | None = None,
    ) -> None:
        self.newton_solver = (
            newton_solver
            if newton_solver is not None
            else NewtonSolver(
                NewtonConfig(
                    residual_tol=1.0e-08,
                    increment_tol=1.0e-08,
                    max_iterations=20,
                )
            )
        )

    def solve(
        self,
        data: BifurcationJunctionData,
        dt: float,
        x0: np.ndarray | None = None,
        raise_on_failure: bool = True,
    ) -> BifurcationSolution:
        residual = BifurcationJunctionResidual(
            data=data,
            dt=dt,
        )

        if x0 is None:
            x0 = residual.initial_guess()

        result = self.newton_solver.solve(
            residual=residual,
            x0=x0,
            jacobian=residual.jacobian,
            raise_on_failure=raise_on_failure,
        )

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = result.x

        return BifurcationSolution(
            parent=BoundaryState(area=float(A_p), flow_rate=float(Q_p)),
            daughter1=BoundaryState(area=float(A_d1), flow_rate=float(Q_d1)),
            daughter2=BoundaryState(area=float(A_d2), flow_rate=float(Q_d2)),
            newton_result=result,
        )