from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hemo1d.core.newton import (
    NewtonConfig,
    NewtonSolver,
    finite_difference_jacobian,
)
from hemo1d.lumped._capillary_bed_data import EndpointSolveData


_CAPILLARY_NEWTON_CONFIG = NewtonConfig(
    residual_tol=1.0e-10,
    max_iterations=30,
    finite_difference_eps=1.0e-6,
    line_search=True,
    relaxed_residual_tol=1.0e-5,
    use_increment_criterion=False,
)


@dataclass(frozen=True)
class NewtonScales:
    area_scales: np.ndarray
    pressure_scale: float
    flow_scale: float


def newton_scales(
    *,
    data: list[EndpointSolveData],
    pressure_old: float,
    venous_pressure: float,
    venous_resistance: float,
) -> NewtonScales:
    area_scales = np.array(
        [max(abs(float(item.area_n)), 1.0) for item in data],
        dtype=float,
    )
    endpoint_pressures = [
        float(item.physics.pressure(item.area_n)) for item in data
    ]
    endpoint_flows = [
        abs((pressure - pressure_old) / item.resistance)
        for item, pressure in zip(data, endpoint_pressures, strict=True)
    ]

    pressure_scale = max(
        abs(float(pressure_old)),
        abs(float(venous_pressure)),
        *(abs(pressure) for pressure in endpoint_pressures),
        1.0,
    )
    flow_scale = max(
        1.0,
        abs((float(pressure_old) - float(venous_pressure)) / venous_resistance),
        *endpoint_flows,
    )

    return NewtonScales(
        area_scales=area_scales,
        pressure_scale=float(pressure_scale),
        flow_scale=float(flow_scale),
    )


def solve_newton(
    *,
    bed_id: str,
    y0: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    scales: NewtonScales,
) -> np.ndarray:
    scale = max(scales.flow_scale, 1.0e-30)
    variable_scales = _variable_scales(scales)

    def scaled_residual_norm(value: np.ndarray) -> float:
        return float(np.linalg.norm(value, ord=np.inf) / scale)

    def jacobian(trial: np.ndarray) -> np.ndarray:
        return finite_difference_jacobian(
            residual=residual,
            x=trial,
            eps=_CAPILLARY_NEWTON_CONFIG.finite_difference_eps,
            variable_scales=variable_scales,
            method="forward",
        )

    result = NewtonSolver(_CAPILLARY_NEWTON_CONFIG).solve(
        residual=residual,
        x0=y0,
        jacobian=jacobian,
        residual_norm=scaled_residual_norm,
        is_valid_candidate=_has_positive_area_unknowns,
    )
    if result.converged:
        return result.x

    raw_residual = float(np.linalg.norm(residual(result.x), ord=np.inf))
    raise RuntimeError(
        f"Capillary bed {bed_id!r} Newton solve failed: "
        f"{result.message} scaled residual {result.residual_norm:.6e}, "
        f"raw infinity residual {raw_residual:.6e}."
    )


def _variable_scales(scales: NewtonScales) -> np.ndarray:
    return np.concatenate(
        [
            scales.area_scales,
            np.array([scales.pressure_scale], dtype=float),
        ]
    )


def _has_positive_area_unknowns(y: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(y)) and np.all(y[:-1] > 0.0))
