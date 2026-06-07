from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, StateAtPoint


DGFluxScheme = Literal["lxf", "hll"]

_DG_FLUX_SCHEME_ALIASES: dict[str, DGFluxScheme] = {
    "lxf": "lxf",
    "lax_friedrichs": "lxf",
    "lax-friedrichs": "lxf",
    "rusanov": "lxf",
    "hll": "hll",
}


def canonicalize_dg_flux_scheme(value: str) -> DGFluxScheme:
    """
    Convert a user-facing DG flux name to its canonical solver value.
    """
    if not isinstance(value, str):
        raise ValueError(
            "DG flux scheme must be a string: accepted values are "
            "'lxf' and 'hll'."
        )

    key = value.strip().lower()
    try:
        return _DG_FLUX_SCHEME_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "Invalid DG flux scheme "
            f"{value!r}; expected 'lxf' or 'hll' "
            "(aliases: 'lax_friedrichs', 'lax-friedrichs', 'rusanov')."
        ) from exc


@dataclass(frozen=True)
class ConservativeState:
    """
    Conservative state U = [A, Q]^T at one point or trace.

    area:
        Cross-sectional area A.

    flow_rate:
        Volumetric flow rate Q.
    """

    area: float
    flow_rate: float

    @classmethod
    def from_point(cls, state: StateAtPoint | BoundaryState) -> ConservativeState:
        return cls(
            area=float(state.area),
            flow_rate=float(state.flow_rate),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.area, self.flow_rate], dtype=np.float64)


def physical_flux(
    physics: Hemo1DPhysics,
    state: ConservativeState,
) -> np.ndarray:
    """
    Conservative physical flux F(U).

    For U = [A, Q]^T:

        F_1 = Q
        F_2 = alpha Q^2/A + C1(A)

    This calls the shared core physics implementation, so the formula remains
    consistent with CG and all boundary/junction logic.
    """
    _validate_state(state)

    flux = physics.flux(state.area, state.flow_rate)

    return np.array(
        [
            float(flux[0]),
            float(flux[1]),
        ],
        dtype=np.float64,
    )


def source_term(
    physics: Hemo1DPhysics,
    state: ConservativeState,
) -> np.ndarray:
    """
    Source term S(U).

    For the current constant-A0, constant-beta model:

        S_1 = 0
        S_2 = K_r Q/A

    The PDE convention is:

        U_t + F(U)_z + S(U) = 0

    so an explicit update subtracts dt * S(U).
    """
    _validate_state(state)

    source = physics.source(state.area, state.flow_rate)

    return np.array(
        [
            float(source[0]),
            float(source[1]),
        ],
        dtype=np.float64,
    )


def max_abs_characteristic_speed(
    physics: Hemo1DPhysics,
    state: ConservativeState,
) -> float:
    """
    Maximum absolute eigenvalue at one state.

    The quasi-linear eigenvalues are:

        lambda_plus  = alpha u + c_alpha
        lambda_minus = alpha u - c_alpha
    """
    _validate_state(state)

    lambda_plus, lambda_minus = physics.eigenvalues(
        state.area,
        state.flow_rate,
    )

    return max(abs(float(lambda_plus)), abs(float(lambda_minus)))


def lax_friedrichs_flux(
    physics: Hemo1DPhysics,
    left: ConservativeState,
    right: ConservativeState,
) -> np.ndarray:
    """
    Local Lax-Friedrichs / Rusanov numerical flux.

    This returns the oriented flux through an interface whose normal points in
    the positive z direction.

        F*(U_L, U_R)
          = 0.5 * (F(U_L) + F(U_R))
            - 0.5 * s_max * (U_R - U_L)

    where:

        s_max = max rho(H(U_L)), rho(H(U_R))

    This is the standard finite-volume/DG interface flux for:

        U_t + F(U)_z + S(U) = 0

    Important orientation convention:
        This flux is positive in the increasing-z direction. The DG0 update is:

            U_j^{n+1} = U_j^n
                        - dt/h * (F*_{j+1/2} - F*_{j-1/2})
                        - dt * S(U_j^n)
    """
    _validate_state(left)
    _validate_state(right)

    F_left = physical_flux(physics, left)
    F_right = physical_flux(physics, right)

    U_left = left.as_array()
    U_right = right.as_array()

    s_max = max(
        max_abs_characteristic_speed(physics, left),
        max_abs_characteristic_speed(physics, right),
    )

    return 0.5 * (F_left + F_right) - 0.5 * s_max * (U_right - U_left)


def hll_flux(
    physics: Hemo1DPhysics,
    left: ConservativeState,
    right: ConservativeState,
) -> np.ndarray:
    """
    Harten-Lax-van Leer numerical flux.

    This returns the oriented flux through an interface whose normal points in
    the positive z direction, matching ``lax_friedrichs_flux``.

    If the HLL denominator is degenerate or the resulting flux is non-finite,
    the local Lax-Friedrichs/Rusanov flux is used as a robust fallback.
    """
    _validate_state(left)
    _validate_state(right)

    F_left = physical_flux(physics, left)
    F_right = physical_flux(physics, right)

    lambda_plus_left, lambda_minus_left = physics.eigenvalues(
        left.area,
        left.flow_rate,
    )
    lambda_plus_right, lambda_minus_right = physics.eigenvalues(
        right.area,
        right.flow_rate,
    )

    s_L = min(float(lambda_minus_left), float(lambda_minus_right))
    s_R = max(float(lambda_plus_left), float(lambda_plus_right))

    if not (np.isfinite(s_L) and np.isfinite(s_R)):
        return lax_friedrichs_flux(physics, left, right)

    if 0.0 <= s_L:
        return F_left
    if s_R <= 0.0:
        return F_right

    denom = s_R - s_L
    scale = max(1.0, abs(s_L), abs(s_R))
    if not np.isfinite(denom) or abs(denom) <= np.finfo(np.float64).eps * scale:
        return lax_friedrichs_flux(physics, left, right)

    U_left = left.as_array()
    U_right = right.as_array()

    flux = (
        s_R * F_left
        - s_L * F_right
        + s_L * s_R * (U_right - U_left)
    ) / denom

    if not np.all(np.isfinite(flux)):
        return lax_friedrichs_flux(physics, left, right)

    return flux


def _validate_state(state: ConservativeState) -> None:
    if state.area <= 0.0:
        raise ValueError(f"Area must be positive, got {state.area}.")
    if not np.isfinite(state.area):
        raise ValueError(f"Area must be finite, got {state.area}.")
    if not np.isfinite(state.flow_rate):
        raise ValueError(f"Flow rate must be finite, got {state.flow_rate}.")
