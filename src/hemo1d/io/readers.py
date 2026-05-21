from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


OutOfBoundsMode = Literal["error", "constant", "periodic", "clamp", "zero"]


@dataclass(frozen=True)
class CSVScalarFunction:
    """Interpolated scalar time series loaded from CSV."""

    time: np.ndarray
    values: np.ndarray
    source_path: Path
    out_of_bounds: OutOfBoundsMode = "error"
    ramp_time: float = 0.0
    ramp_kind: str = "cosine"

    def __call__(self, t: float) -> float:
        value = self._interpolate(float(t))
        return ramp_factor(float(t), self.ramp_time, kind=self.ramp_kind) * value

    @property
    def t_min(self) -> float:
        return float(self.time[0])

    @property
    def t_max(self) -> float:
        return float(self.time[-1])

    @property
    def period(self) -> float:
        return self.t_max - self.t_min

    def _interpolate(self, t: float) -> float:
        mode = _normalize_out_of_bounds(self.out_of_bounds)

        if mode == "periodic":
            if self.period <= 0.0:
                raise ValueError("Cannot use periodic CSV input with non-positive period.")
            t_eval = ((t - self.t_min) % self.period) + self.t_min
            return float(np.interp(t_eval, self.time, self.values))

        if mode == "error":
            if t < self.t_min or t > self.t_max:
                raise ValueError(
                    f"Time {t} outside CSV interval [{self.t_min}, {self.t_max}] "
                    f"for {self.source_path}."
                )
            return float(np.interp(t, self.time, self.values))

        if mode == "zero":
            if t < self.t_min or t > self.t_max:
                return 0.0
            return float(np.interp(t, self.time, self.values))

        return float(
            np.interp(
                t,
                self.time,
                self.values,
                left=self.values[0],
                right=self.values[-1],
            )
        )


@dataclass(frozen=True)
class VelocityInflowSeries(CSVScalarFunction):
    """Backward-compatible velocity inflow series loaded from CSV."""

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        time_column: str = "Time",
        velocity_column: str = "Velocity",
        velocity_scale_to_cm_per_s: float = 1.0,
        shift_time_to_zero: bool = True,
        out_of_bounds: OutOfBoundsMode = "periodic",
    ) -> "VelocityInflowSeries":
        series = _read_scalar_csv(
            path,
            time_column=time_column,
            value_column=velocity_column,
            scale=velocity_scale_to_cm_per_s,
            out_of_bounds=out_of_bounds,
            ramp_time=0.0,
            ramp_kind="cosine",
            shift_time_to_zero=shift_time_to_zero,
            cls=cls,
        )
        assert isinstance(series, cls)
        return series

    @property
    def velocity_cm_s(self) -> np.ndarray:
        return self.values

    def velocity(self, t: float) -> float:
        return self(t)

    def flow_rate(self, t: float, area0: float) -> float:
        return area0 * self.velocity(t)


def read_velocity_csv(
    path: str | Path,
    *,
    time_column: str = "Time",
    value_column: str = "Velocity",
    velocity_column: str | None = None,
    scale: float = 1.0,
    out_of_bounds: OutOfBoundsMode = "error",
    ramp_time: float = 0.0,
    ramp_kind: str = "cosine",
    shift_time_to_zero: bool = True,
) -> Callable[[float], float]:
    """Read a velocity time series from CSV and return ``v(t)``."""

    return _read_scalar_csv(
        path,
        time_column=time_column,
        value_column=velocity_column or value_column,
        scale=scale,
        out_of_bounds=out_of_bounds,
        ramp_time=ramp_time,
        ramp_kind=ramp_kind,
        shift_time_to_zero=shift_time_to_zero,
        cls=CSVScalarFunction,
    )


def read_flow_rate_csv(
    path: str | Path,
    *,
    time_column: str = "Time",
    value_column: str = "FlowRate",
    flow_rate_column: str | None = None,
    scale: float = 1.0,
    out_of_bounds: OutOfBoundsMode = "error",
    ramp_time: float = 0.0,
    ramp_kind: str = "cosine",
    shift_time_to_zero: bool = True,
) -> Callable[[float], float]:
    """Read a flow-rate time series from CSV and return ``Q(t)``."""

    return _read_scalar_csv(
        path,
        time_column=time_column,
        value_column=flow_rate_column or value_column,
        scale=scale,
        out_of_bounds=out_of_bounds,
        ramp_time=ramp_time,
        ramp_kind=ramp_kind,
        shift_time_to_zero=shift_time_to_zero,
        cls=CSVScalarFunction,
    )


def read_area_csv(
    path: str | Path,
    *,
    time_column: str = "Time",
    value_column: str = "Area",
    area_column: str | None = None,
    scale: float = 1.0,
    out_of_bounds: OutOfBoundsMode = "error",
    ramp_time: float = 0.0,
    ramp_kind: str = "cosine",
    shift_time_to_zero: bool = True,
) -> Callable[[float], float]:
    """Read an area time series from CSV and return ``A(t)``."""

    return _read_scalar_csv(
        path,
        time_column=time_column,
        value_column=area_column or value_column,
        scale=scale,
        out_of_bounds=out_of_bounds,
        ramp_time=ramp_time,
        ramp_kind=ramp_kind,
        shift_time_to_zero=shift_time_to_zero,
        cls=CSVScalarFunction,
    )


def ramp_factor(t: float, ramp_time: float, *, kind: str = "cosine") -> float:
    """Return a multiplicative ramp from 0 to 1 over ``ramp_time``."""

    if ramp_time < 0.0:
        raise ValueError("ramp_time must be non-negative.")
    if ramp_time <= 0.0:
        return 1.0

    x = t / ramp_time
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if kind == "linear":
        return float(x)
    if kind == "cosine":
        return float(0.5 * (1.0 - np.cos(np.pi * x)))

    raise ValueError("ramp kind must be one of: 'linear', 'cosine'.")


def _read_scalar_csv(
    path: str | Path,
    *,
    time_column: str,
    value_column: str,
    scale: float,
    out_of_bounds: OutOfBoundsMode,
    ramp_time: float,
    ramp_kind: str,
    shift_time_to_zero: bool,
    cls: type[CSVScalarFunction] = CSVScalarFunction,
) -> CSVScalarFunction:
    path = Path(path)
    _validate_out_of_bounds(out_of_bounds)
    _ = ramp_factor(0.0, ramp_time, kind=ramp_kind)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.dtype.names is None:
        raise ValueError(f"Expected named CSV columns in {path}.")

    time_name = _find_column(data.dtype.names, time_column)
    value_name = _find_column(data.dtype.names, value_column)

    time = np.asarray(data[time_name], dtype=float)
    values = np.asarray(data[value_name], dtype=float)
    if time.ndim == 0:
        time = np.array([float(time)])
        values = np.array([float(values)])

    if len(time) < 2:
        raise ValueError(f"CSV file {path} must contain at least two rows.")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"Time column in {path} contains non-finite values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Value column in {path} contains non-finite values.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError(f"Time column in {path} must be strictly increasing.")

    if shift_time_to_zero:
        time = time - time[0]

    return cls(
        time=time,
        values=scale * values,
        source_path=path,
        out_of_bounds=out_of_bounds,
        ramp_time=ramp_time,
        ramp_kind=ramp_kind,
    )


def _find_column(columns: tuple[str, ...], requested: str) -> str:
    if requested in columns:
        return requested

    normalized = {column.lower(): column for column in columns}
    key = requested.lower()
    if key in normalized:
        return normalized[key]

    raise ValueError(f"Column {requested!r} not found. Available columns: {columns}")


def _normalize_out_of_bounds(mode: OutOfBoundsMode) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "clamp":
        return "constant"
    return normalized


def _validate_out_of_bounds(mode: OutOfBoundsMode) -> None:
    if _normalize_out_of_bounds(mode) not in {"error", "constant", "periodic", "zero"}:
        raise ValueError(
            "out_of_bounds must be one of: 'error', 'constant', 'periodic'."
        )
