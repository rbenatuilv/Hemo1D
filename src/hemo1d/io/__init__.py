from __future__ import annotations

from pathlib import Path
from typing import Callable

from hemo1d.io.config_loader import load_json
from hemo1d.io.readers import (
    CSVScalarFunction,
    VelocityInflowSeries,
    read_area_csv,
    read_flow_rate_csv,
    read_velocity_csv,
    ramp_factor,
)
from hemo1d.io.writers import (
    write_diagnostics_csv,
    write_probe_history_csv,
    write_vessel_final_state_csv,
)

def make_flow_rate_from_velocity_csv(
    *,
    vessel_id: str,
    area0: float,
    csv_path: Path,
    time_column: str = "Time",
    velocity_column: str = "Velocity",
    velocity_scale_to_cm_per_s: float = 1.0,
    out_of_bounds: str = "constant",
    ramp_time: float = 0.0,
    ramp_kind: str = "cosine",
) -> Callable[[float], float]:
    """Backward-compatible helper that returns ``Q(t) = A0 * v(t)``."""

    if area0 <= 0.0:
        raise ValueError("area0 must be positive.")

    velocity = read_velocity_csv(
        csv_path,
        time_column=time_column,
        velocity_column=velocity_column,
        scale=velocity_scale_to_cm_per_s,
        out_of_bounds=out_of_bounds,  # type: ignore[arg-type]
        ramp_time=ramp_time,
        ramp_kind=ramp_kind,
    )
    return lambda t: area0 * velocity(t)


__all__ = [
    "CSVScalarFunction",
    "VelocityInflowSeries",
    "load_json",
    "make_flow_rate_from_velocity_csv",
    "ramp_factor",
    "read_area_csv",
    "read_flow_rate_csv",
    "read_velocity_csv",
    "write_diagnostics_csv",
    "write_probe_history_csv",
    "write_vessel_final_state_csv",
]
