from __future__ import annotations

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide


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


__all__ = [
    "compatibility_target",
    "outgoing_left_eigenvector",
]
