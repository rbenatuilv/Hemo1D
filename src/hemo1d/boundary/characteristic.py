from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hemo1d.boundary.base import BoundaryCondition
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide


ScalarFunction = Callable[[float], float]


def _as_vector(area: float, flow_rate: float) -> np.ndarray:
    return np.array([area, flow_rate], dtype=float)


def _compatibility_target(
    physics: Hemo1DPhysics,
    endpoint_data: EndpointData,
    dt: float,
) -> np.ndarray:
    """
    Compute:

        CC = U^n - dt * H(U^n) dU^n/dz - dt * S(U^n)

    This is the explicit first-order compatibility update used at boundaries.
    """
    A = endpoint_data.state.area
    Q = endpoint_data.state.flow_rate

    U = _as_vector(A, Q)
    dU_dz = _as_vector(endpoint_data.d_area_dz, endpoint_data.d_flow_rate_dz)

    H = physics.H_matrix(A, Q)
    S = physics.source(A, Q)

    return U - dt * (H @ dU_dz + S)


def _non_reflecting_target(
    physics: Hemo1DPhysics,
    endpoint_data: EndpointData,
    dt: float,
) -> np.ndarray:
    """
    Compute:

        NR = U^n - dt * S(U^n)

    This corresponds to the non-reflecting condition:

        l^T (dU/dt + S) = 0.
    """
    A = endpoint_data.state.area
    Q = endpoint_data.state.flow_rate

    U = _as_vector(A, Q)
    S = physics.source(A, Q)

    return U - dt * S


def _outgoing_left_eigenvector(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    side: EndpointSide,
) -> tuple[float, float]:
    """
    Return the outgoing-characteristic left eigenvector at the endpoint.

    Under subcritical flow:

        lambda_plus  > 0
        lambda_minus < 0

    At LEFT endpoint z=0:
        outgoing from domain is lambda_minus.

    At RIGHT endpoint z=L:
        outgoing from domain is lambda_plus.
    """
    l_plus, l_minus = physics.left_eigenvectors(area, flow_rate)

    if side == EndpointSide.LEFT:
        return l_minus
    if side == EndpointSide.RIGHT:
        return l_plus

    raise ValueError(f"Unknown endpoint side: {side}")


def _incoming_left_eigenvector(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    side: EndpointSide,
) -> tuple[float, float]:
    """
    Return the incoming-characteristic left eigenvector at the endpoint.

    Under subcritical flow:

        at LEFT endpoint: incoming is lambda_plus
        at RIGHT endpoint: incoming is lambda_minus
    """
    l_plus, l_minus = physics.left_eigenvectors(area, flow_rate)

    if side == EndpointSide.LEFT:
        return l_plus
    if side == EndpointSide.RIGHT:
        return l_minus

    raise ValueError(f"Unknown endpoint side: {side}")


class PrescribedFlowBoundary(BoundaryCondition):
    """
    Boundary condition prescribing flow rate Q(t).

    The other unknown, area A, is computed from the compatibility equation
    associated with the outgoing characteristic.

    This can be used at either endpoint, although for the first solver we will
    mostly use it at the inlet.
    """

    def __init__(self, flow_rate: ScalarFunction) -> None:
        self.flow_rate = flow_rate

    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        A_n = endpoint_data.state.area
        Q_n = endpoint_data.state.flow_rate

        q_value = float(self.flow_rate(t))
        if side == EndpointSide.RIGHT:
            q_value = -q_value

        CC = _compatibility_target(physics, endpoint_data, dt)
        l_out = _outgoing_left_eigenvector(physics, A_n, Q_n, side)

        lA = float(l_out[0])
        lQ = float(l_out[1])

        if abs(lA) < 1.0e-14:
            raise RuntimeError("Cannot solve prescribed-flow boundary: lA is too small.")

        target = float(lA * CC[0] + lQ * CC[1])

        area = (target - lQ * q_value) / lA

        return BoundaryState(
            area=area,
            flow_rate=q_value,
        )


class PrescribedAreaBoundary(BoundaryCondition):
    """
    Boundary condition prescribing area A(t).

    The other unknown, flow Q, is computed from the compatibility equation.
    """

    def __init__(self, area: ScalarFunction) -> None:
        self.area = area

    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        A_n = endpoint_data.state.area
        Q_n = endpoint_data.state.flow_rate

        area_value = float(self.area(t))

        CC = _compatibility_target(physics, endpoint_data, dt)
        l_out = _outgoing_left_eigenvector(physics, A_n, Q_n, side)

        lA = float(l_out[0])
        lQ = float(l_out[1])

        if abs(lQ) < 1.0e-14:
            raise RuntimeError("Cannot solve prescribed-area boundary: lQ is too small.")

        target = float(lA * CC[0] + lQ * CC[1])

        flow_rate = (target - lA * area_value) / lQ

        return BoundaryState(
            area=area_value,
            flow_rate=flow_rate,
        )


class PrescribedPressureBoundary(BoundaryCondition):
    """
    Boundary condition prescribing internal pressure P(t).

    The pressure is converted to area using the existing tube law:

        P = p_ext + p0 + beta (sqrt(A) - sqrt(A0)) / A0

    The remaining unknown, Q, is computed with the same outgoing-characteristic
    compatibility equation used by ``PrescribedAreaBoundary``.
    """

    def __init__(self, pressure: ScalarFunction) -> None:
        self.pressure = pressure

    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        pressure_value = float(self.pressure(t))
        area_value = _area_from_pressure(physics, pressure_value)

        return PrescribedAreaBoundary(lambda _t: area_value).compute(
            physics=physics,
            endpoint_data=endpoint_data,
            side=side,
            t=t,
            dt=dt,
        )


class NonReflectingBoundary(BoundaryCondition):
    """
    Non-reflecting boundary condition.

    It combines:

        1. compatibility equation along the outgoing characteristic;
        2. non-reflecting equation along the incoming characteristic.

    The result is a 2x2 linear system for:

        A^{n+1}, Q^{n+1}.
    """

    def compute(
        self,
        physics: Hemo1DPhysics,
        endpoint_data: EndpointData,
        side: EndpointSide,
        t: float,
        dt: float,
    ) -> BoundaryState:
        A_n = endpoint_data.state.area
        Q_n = endpoint_data.state.flow_rate

        CC = _compatibility_target(physics, endpoint_data, dt)
        NR = _non_reflecting_target(physics, endpoint_data, dt)

        l_out = _outgoing_left_eigenvector(physics, A_n, Q_n, side)
        l_in = _incoming_left_eigenvector(physics, A_n, Q_n, side)

        matrix = np.array(
            [
                [float(l_out[0]), float(l_out[1])],
                [float(l_in[0]), float(l_in[1])],
            ],
            dtype=float,
        )

        rhs = np.array(
            [
                float(l_out[0] * CC[0] + l_out[1] * CC[1]),
                float(l_in[0] * NR[0] + l_in[1] * NR[1]),
            ],
            dtype=float,
        )

        area, flow_rate = np.linalg.solve(matrix, rhs)

        return BoundaryState(
            area=float(area),
            flow_rate=float(flow_rate),
        )


def _area_from_pressure(physics: Hemo1DPhysics, pressure: float) -> float:
    A0 = physics.params.area0
    beta = physics.params.beta
    root_area = np.sqrt(A0) + A0 * (
        pressure - physics.params.p_ext - physics.params.p0
    ) / beta

    if root_area <= 0.0:
        raise ValueError(
            "Prescribed pressure gives non-positive sqrt(area): "
            f"P={pressure}, sqrt(A)={root_area}."
        )

    return float(root_area * root_area)
