from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from hemo1d.convergence.errors import linf_time_l2_space_error
from hemo1d.convergence.richardson import richardson_extrapolate_history
from hemo1d.convergence.snapshots import SnapshotHistory


@dataclass(frozen=True)
class ConvergenceLevel:
    """
    One discretization level for a convergence study.
    """

    name: str
    num_cells: int
    dt: float
    h: float | None = None

    @property
    def h_like(self) -> float:
        if self.h is not None:
            return self.h
        return 1.0 / self.num_cells


@dataclass(frozen=True)
class ConvergenceErrorRow:
    """
    Error row for one level.
    """

    level_name: str
    num_cells: int
    dt: float
    h_like: float
    area_error: float
    flow_rate_error: float


def compute_errors_against_reference(
    level_histories: dict[str, SnapshotHistory],
    levels: list[ConvergenceLevel],
    reference_name: str,
) -> list[ConvergenceErrorRow]:
    """
    Compute L∞(time; L2(space)) errors against a reference history.

    All histories must be sampled on the same spatial grid and recorded at
    compatible times.
    """
    reference = level_histories[reference_name]

    rows: list[ConvergenceErrorRow] = []

    for level in levels:
        if level.name == reference_name:
            continue

        history = level_histories[level.name]

        area_error = linf_time_l2_space_error(
            solution=history,
            reference=reference,
            field="area",
        )
        flow_error = linf_time_l2_space_error(
            solution=history,
            reference=reference,
            field="flow_rate",
        )

        rows.append(
            ConvergenceErrorRow(
                level_name=level.name,
                num_cells=level.num_cells,
                dt=level.dt,
                h_like=level.h_like,
                area_error=area_error,
                flow_rate_error=flow_error,
            )
        )

    return rows


def compute_richardson_error_rows(
    level_histories: dict[str, SnapshotHistory],
    levels: list[ConvergenceLevel],
    expected_order: float = 2.0,
    delta: float = 2.0,
) -> list[ConvergenceErrorRow]:
    """
    Compute Richardson-estimated errors for consecutive level pairs.

    For each pair (coarse, fine), build:

        R = (delta^p * fine - coarse) / (delta^p - 1)

    and report:

        ||R - coarse||_{L∞(L2)}

    This estimates the error of the coarse solution.
    """
    rows: list[ConvergenceErrorRow] = []

    for coarse_level, fine_level in zip(levels[:-1], levels[1:]):
        coarse = level_histories[coarse_level.name]
        fine = level_histories[fine_level.name]

        rich = richardson_extrapolate_history(
            coarse=coarse,
            fine=fine,
            expected_order=expected_order,
            delta=delta,
        )

        area_error = linf_time_l2_space_error(
            solution=coarse,
            reference=rich,
            field="area",
        )
        flow_error = linf_time_l2_space_error(
            solution=coarse,
            reference=rich,
            field="flow_rate",
        )

        rows.append(
            ConvergenceErrorRow(
                level_name=coarse_level.name,
                num_cells=coarse_level.num_cells,
                dt=coarse_level.dt,
                h_like=coarse_level.h_like,
                area_error=area_error,
                flow_rate_error=flow_error,
            )
        )

    return rows


def write_convergence_errors_csv(
    rows: list,
    output_path: str | Path,
) -> None:
    """
    Write convergence error rows to CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "level_name",
        "num_cells",
        "dt",
        "h_like",
        "area_error",
        "flow_rate_error",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "level_name": row.level_name,
                    "num_cells": row.num_cells,
                    "dt": row.dt,
                    "h_like": row.h_like,
                    "area_error": row.area_error,
                    "flow_rate_error": row.flow_rate_error,
                }
            )
