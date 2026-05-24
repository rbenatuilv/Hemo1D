from __future__ import annotations

import numpy as np

from hemo1d.core.state import BoundaryState
from hemo1d.lumped._capillary_bed_data import EndpointSolveData
from hemo1d.topology.endpoint import NetworkEndpoint


def initial_guess(
    data: list[EndpointSolveData],
    pressure_old: float,
) -> np.ndarray:
    y = np.empty(len(data) + 1, dtype=float)
    for i, item in enumerate(data):
        y[i] = item.area_n
    y[-1] = pressure_old
    return y


def residual(
    y: np.ndarray,
    *,
    data: list[EndpointSolveData],
    compliance: float,
    venous_pressure: float,
    venous_resistance: float,
    pressure_old: float,
    dt: float,
) -> np.ndarray:
    n = len(data)
    value = np.empty(n + 1, dtype=float)

    if not np.all(np.isfinite(y)) or np.any(y[:n] <= 0.0):
        return np.full(n + 1, 1.0e30, dtype=float)

    pressure = float(y[-1])
    total_inflow = 0.0

    for i, item in enumerate(data):
        area = float(y[i])
        flow_rate = item.flow_rate(area)
        inflow = item.endpoint.outward_flow(flow_rate)
        pressure_1d = float(item.physics.pressure(area))

        value[i] = inflow - (pressure_1d - pressure) / item.resistance
        total_inflow += inflow

    value[-1] = (
        compliance * (pressure - pressure_old) / dt
        - total_inflow
        + venous_outflow(
            pressure,
            venous_pressure=venous_pressure,
            venous_resistance=venous_resistance,
        )
    )

    if not np.all(np.isfinite(value)):
        return np.full(n + 1, 1.0e30, dtype=float)

    return value


def endpoint_states_from_solution(
    y: np.ndarray,
    data: list[EndpointSolveData],
) -> tuple[dict[NetworkEndpoint, BoundaryState], dict[NetworkEndpoint, float]]:
    endpoint_states: dict[NetworkEndpoint, BoundaryState] = {}
    endpoint_inflows: dict[NetworkEndpoint, float] = {}

    for i, item in enumerate(data):
        area = float(y[i])
        flow_rate = item.flow_rate(area)
        endpoint_states[item.endpoint] = BoundaryState(
            area=area,
            flow_rate=flow_rate,
        )
        endpoint_inflows[item.endpoint] = item.endpoint.outward_flow(flow_rate)

    return endpoint_states, endpoint_inflows


def venous_outflow(
    pressure: float,
    *,
    venous_pressure: float,
    venous_resistance: float,
) -> float:
    return float((pressure - venous_pressure) / venous_resistance)
