from __future__ import annotations

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics


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
    """Pressure loss term for a junction with angle/losses."""
    if angle is None:
        return 0.0

    gamma = physics.params.gamma_pressure
    if gamma == 0.0:
        return 0.0

    angle_factor = np.sqrt(2.0 * (1.0 - np.cos(angle)))
    return gamma * flow_rate * abs(flow_rate) / area**2 * angle_factor


def _d_pressure_loss_term_gradient(
    physics: Hemo1DPhysics,
    area: float,
    flow_rate: float,
    angle: float | None,
) -> tuple[float, float]:
    """Gradient of the pressure loss term with respect to (A, Q)."""
    if angle is None:
        return 0.0, 0.0

    gamma = physics.params.gamma_pressure
    if gamma == 0.0:
        return 0.0, 0.0

    angle_factor = np.sqrt(2.0 * (1.0 - np.cos(angle)))
    common_factor = gamma * abs(flow_rate) / area**2 * angle_factor

    d_loss_dA = -2.0 * common_factor * flow_rate / area
    d_loss_dQ = 2.0 * common_factor

    return d_loss_dA, d_loss_dQ
