from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hemo1d.builder import SolverSettings
from hemo1d.config import NetworkConfig
from hemo1d.results import Results

if TYPE_CHECKING:
    from hemo1d.convergence.reports import ConvergenceErrorRow


@dataclass(frozen=True)
class ConvergenceStudyLevel:
    name: str
    h: float
    dt: float
    num_cells: int
    result: Results


@dataclass
class ConvergenceStudy:
    """Whole-network convergence-study result for the public facade."""

    levels: list[ConvergenceStudyLevel]
    expected_order: float
    refinement_ratio: float = 2.0
    _error_rows_cache: list[ConvergenceErrorRow] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def error_rows(self) -> list[ConvergenceErrorRow]:
        """Richardson-estimated full-solution errors for each coarse level."""

        from hemo1d.convergence.network_errors import (
            compute_network_richardson_error_rows,
        )
        from hemo1d.convergence.reports import ConvergenceLevel

        if self._error_rows_cache is None:
            histories = {
                level.name: level.result.history.snapshots
                for level in self.levels
            }
            missing = [
                name
                for name, history in histories.items()
                if not history.snapshots
            ]
            if missing:
                raise ValueError(
                    "Convergence levels do not contain spatial snapshots: "
                    f"{missing}."
                )

            convergence_levels = [
                ConvergenceLevel(
                    name=level.name,
                    num_cells=level.num_cells,
                    dt=level.dt,
                    h=level.h,
                )
                for level in self.levels
            ]
            self._error_rows_cache = compute_network_richardson_error_rows(
                level_histories=histories,
                levels=convergence_levels,
                expected_order=self.expected_order,
                delta=self.refinement_ratio,
            )

        return list(self._error_rows_cache)

    @property
    def observed_orders(self) -> dict[str, list[float]]:
        from hemo1d.convergence.network_errors import network_observed_orders

        area_orders, flow_orders = network_observed_orders(
            self.error_rows,
            delta=self.refinement_ratio,
        )
        return {
            "area": area_orders,
            "flow_rate": flow_orders,
        }

    def save(self, path: str | Path) -> None:
        """Save convergence summary and level outputs."""

        from hemo1d.convergence.reports import write_convergence_errors_csv

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        write_convergence_errors_csv(
            self.error_rows,
            output_dir / "convergence.csv",
        )

        for level in self.levels:
            level.result.save(output_dir / level.name)

    def plot(self, output_dir: str | Path | None = None, *, show: bool = True) -> None:
        """Plot full-solution area and flow-rate convergence errors."""

        from hemo1d.convergence.plotting import plot_convergence_errors_separate

        rows = self.error_rows
        if (
            not rows
            or any(row.area_error <= 0.0 for row in rows)
            or any(row.flow_rate_error <= 0.0 for row in rows)
        ):
            return

        plot_convergence_errors_separate(
            rows=rows,
            output_dir=output_dir,
            filename_prefix="convergence",
            title_prefix="Full-solution convergence",
            reference_orders=(2.0, 1.0, 0.5),
            show=show,
        )


def validate_refinement_ratio(h_levels: list[float]) -> float:
    if any(h <= 0.0 for h in h_levels):
        raise ValueError("h_levels must all be positive.")

    ratios: list[float] = []
    for coarse_h, fine_h in zip(h_levels[:-1], h_levels[1:]):
        if fine_h >= coarse_h:
            raise ValueError("h_levels must be strictly decreasing.")
        ratios.append(float(coarse_h / fine_h))

    refinement_ratio = ratios[0]
    for ratio in ratios[1:]:
        if not np.isclose(ratio, refinement_ratio, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("h_levels must use a constant refinement ratio.")

    return refinement_ratio


def snapshot_sample_points_by_vessel(
    *,
    config: NetworkConfig,
    h: float,
) -> dict[str, np.ndarray]:
    return {
        vessel_id: np.linspace(
            0.0,
            vessel.length,
            max(1, int(np.ceil(vessel.length / h))) + 1,
        )
        for vessel_id, vessel in config.vessels.items()
    }


def network_num_cells(result: Results) -> int:
    total = 0
    for vessel in result.network.vessels.values():
        num_cells = getattr(vessel.discretization, "num_cells", None)
        if num_cells is None:
            raise ValueError(
                f"Cannot determine num_cells for vessel {vessel.vessel_id!r}."
            )
        total += int(num_cells)
    return total


def convergence_progress_description(
    *,
    level_name: str,
    level_index: int,
    total_levels: int,
    settings: SolverSettings,
    h: float,
    dt: float,
    t_end: float,
) -> str:
    return (
        f"Convergence {level_name} ({level_index + 1}/{total_levels}): "
        f"method={settings.method}, poly_order={settings.poly_order}, "
        f"h={h:g}, dt={dt:g}, t_end={t_end:g}"
    )


__all__ = [
    "ConvergenceStudy",
    "ConvergenceStudyLevel",
    "convergence_progress_description",
    "network_num_cells",
    "snapshot_sample_points_by_vessel",
    "validate_refinement_ratio",
]
