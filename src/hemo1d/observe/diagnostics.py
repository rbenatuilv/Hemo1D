from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics


@dataclass(frozen=True)
class StateDiagnostics:
    """
    Basic diagnostics for one vessel state.

    These diagnostics are not meant to replace probes. They are stability and
    sanity-check quantities:

        - minimum and maximum area,
        - minimum and maximum flow rate,
        - maximum pressure,
        - maximum wave speed.
    """

    time: float
    min_area: float
    max_area: float
    min_flow_rate: float
    max_flow_rate: float
    max_pressure: float
    max_wave_speed: float


def compute_state_diagnostics_from_arrays(
    *,
    area: np.ndarray,
    flow_rate: np.ndarray,
    physics: Hemo1DPhysics,
    time: float,
) -> StateDiagnostics:
    """
    Compute diagnostics from raw A and Q arrays.

    This is discretization-agnostic. CG, DG, or any future discretization only
    needs to provide representative arrays for A and Q.
    """
    pressure = physics.pressure(area)
    wave_speed = physics.wave_speed(area)

    return StateDiagnostics(
        time=float(time),
        min_area=float(np.min(area)),
        max_area=float(np.max(area)),
        min_flow_rate=float(np.min(flow_rate)),
        max_flow_rate=float(np.max(flow_rate)),
        max_pressure=float(np.max(pressure)),
        max_wave_speed=float(np.max(wave_speed)),
    )


def compute_vessel_diagnostics(
    vessel: Any,
    time: float,
) -> StateDiagnostics:
    """
    Compute diagnostics for a generic Vessel.

    The vessel must expose:

        vessel.state_arrays() -> (z, A, Q)
        vessel.physics
    """
    _, area, flow_rate = vessel.state_arrays()

    return compute_state_diagnostics_from_arrays(
        area=area,
        flow_rate=flow_rate,
        physics=vessel.physics,
        time=time,
    )


def compute_cg_state_diagnostics(
    state: Any,
    physics: Hemo1DPhysics,
    time: float,
) -> StateDiagnostics:
    """
    Backward-compatible helper for direct CG state diagnostics.

    This remains useful for low-level CG tests, but generic solver code should
    use compute_vessel_diagnostics(...).
    """
    area = state.A.x.array
    flow_rate = state.Q.x.array

    return compute_state_diagnostics_from_arrays(
        area=area,
        flow_rate=flow_rate,
        physics=physics,
        time=time,
    )