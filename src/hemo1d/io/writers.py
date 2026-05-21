from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_diagnostics_csv(
    history: Any,
    path: str | Path,
) -> None:
    """Write diagnostic history to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not history.diagnostics:
        with path.open("w", newline="") as file:
            csv.writer(file).writerow(["time"])
        return

    first = history.diagnostics[0]
    if hasattr(first, "vessel_diagnostics"):
        fieldnames = [
            "time",
            "vessel_id",
            "min_area",
            "max_area",
            "min_flow_rate",
            "max_flow_rate",
            "max_pressure",
            "max_wave_speed",
        ]

        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for sample in history.diagnostics:
                for vessel_id, d in sample.vessel_diagnostics.items():
                    writer.writerow(
                        {
                            "time": sample.time,
                            "vessel_id": vessel_id,
                            "min_area": d.min_area,
                            "max_area": d.max_area,
                            "min_flow_rate": d.min_flow_rate,
                            "max_flow_rate": d.max_flow_rate,
                            "max_pressure": d.max_pressure,
                            "max_wave_speed": d.max_wave_speed,
                        }
                    )
        return

    fieldnames = [
        "time",
        "min_area",
        "max_area",
        "min_flow_rate",
        "max_flow_rate",
        "max_pressure",
        "max_wave_speed",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for d in history.diagnostics:
            writer.writerow(
                {
                    "time": d.time,
                    "min_area": d.min_area,
                    "max_area": d.max_area,
                    "min_flow_rate": d.min_flow_rate,
                    "max_flow_rate": d.max_flow_rate,
                    "max_pressure": d.max_pressure,
                    "max_wave_speed": d.max_wave_speed,
                }
            )


def write_probe_history_csv(
    history: Any,
    path: str | Path,
) -> None:
    """Write probe history to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time",
        "vessel_id",
        "name",
        "coordinate",
        "area",
        "flow_rate",
        "pressure",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in history.probes.samples:
            writer.writerow(
                {
                    "time": sample.time,
                    "vessel_id": sample.vessel_id,
                    "name": sample.name,
                    "coordinate": sample.coordinate,
                    "area": sample.area,
                    "flow_rate": sample.flow_rate,
                    "pressure": sample.pressure,
                }
            )


def write_vessel_final_state_csv(
    vessel: Any,
    path: str | Path,
) -> None:
    """Write a generic vessel final state to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    z, area, flow_rate = vessel.state_arrays()

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["z", "area", "flow_rate"])
        writer.writeheader()
        for zi, ai, qi in zip(z, area, flow_rate):
            writer.writerow(
                {
                    "z": float(zi),
                    "area": float(ai),
                    "flow_rate": float(qi),
                }
            )
