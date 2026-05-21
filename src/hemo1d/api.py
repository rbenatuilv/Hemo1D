from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hemo1d.boundary import (
    BoundaryCondition,
    NonReflectingBoundary,
    PrescribedAreaBoundary,
    PrescribedFlowBoundary,
    PrescribedPressureBoundary,
)
from hemo1d.builder import SolverSettings, build_vascular_network
from hemo1d.config import NetworkConfig, load_network_config, parse_endpoint_side
from hemo1d.core.state import EndpointSide
from hemo1d.io import read_area_csv, read_flow_rate_csv, read_velocity_csv
from hemo1d.topology.endpoint import NetworkEndpoint
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.observe import ProbePoint
from hemo1d.results import Results
from hemo1d.solvers.time import TimeConfig


ScalarFunction = Callable[[float], float]

if TYPE_CHECKING:
    from hemo1d.convergence.reports import ConvergenceErrorRow


@dataclass(frozen=True)
class BoundaryAssignment:
    """Public boundary assignment stored before a solver network is built."""

    endpoint: NetworkEndpoint
    kind: str
    function: ScalarFunction | None = None


class HemodynamicModel:
    """
    High-level user-facing model facade.

    The model stores declarative network, boundary, solver, and probe settings.
    A fresh mutable solver network is built for every ``solve`` call so repeated
    runs and convergence studies do not reuse old state.
    """

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self.solver_settings = SolverSettings()
        self._boundaries: dict[NetworkEndpoint, BoundaryAssignment] = {}
        self._probes: list[ProbePoint] = []
        self._last_result: Results | None = None

        self._install_default_boundaries_from_config()

    @classmethod
    def from_config(cls, path: str | Path) -> HemodynamicModel:
        """Load a model from a JSON network configuration."""

        return cls(load_network_config(path))

    def set_solver(
        self,
        *,
        method: str = "CG",
        h: float | None = None,
        dt: float | None = None,
        poly_order: int = 1,
        num_cells: int | dict[str, int] | None = None,
        cfl: float | None = None,
        dg_time_scheme: str = "rk2",
        record_every: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Configure the spatial method and time-step controls."""

        self.solver_settings = SolverSettings(
            method=method.lower(),  # type: ignore[arg-type]
            h=h,
            dt=dt,
            poly_order=poly_order,
            num_cells=num_cells,
            cfl=self.solver_settings.cfl if cfl is None else cfl,
            dg_time_scheme=dg_time_scheme.lower(),  # type: ignore[arg-type]
            record_every=(
                self.solver_settings.record_every
                if record_every is None
                else record_every
            ),
            max_steps=(
                self.solver_settings.max_steps
                if max_steps is None
                else max_steps
            ),
        )

    def set_inlet(
        self,
        *,
        vessel_id: str,
        kind: str,
        function: ScalarFunction,
        side: EndpointSide | str | None = None,
    ) -> None:
        """Set an external inlet boundary condition for one vessel endpoint."""

        endpoint = self._resolve_external_endpoint(vessel_id, side, role="inlet")
        self._boundaries[endpoint] = BoundaryAssignment(
            endpoint=endpoint,
            kind=_normalize_boundary_kind(kind),
            function=function,
        )

    def set_outlet(
        self,
        *,
        vessel_id: str,
        kind: str = "nonreflecting",
        function: ScalarFunction | None = None,
        side: EndpointSide | str | None = None,
    ) -> None:
        """Set an external outlet boundary condition for one vessel endpoint."""

        endpoint = self._resolve_external_endpoint(vessel_id, side, role="outlet")
        self._boundaries[endpoint] = BoundaryAssignment(
            endpoint=endpoint,
            kind=_normalize_boundary_kind(kind),
            function=function,
        )

    def add_probe(
        self,
        *,
        vessel_id: str,
        position: float,
        name: str | None = None,
    ) -> None:
        """Add a physical-coordinate probe on one vessel."""

        vessel = self.config.vessel(vessel_id)
        if position < 0.0 or position > vessel.length:
            raise ValueError(
                f"Probe position {position} is outside vessel {vessel_id!r} "
                f"domain [0, {vessel.length}]."
            )

        probe_name = name if name is not None else f"probe_{len(self._probes) + 1}"
        self._probes.append(
            ProbePoint(
                vessel_id=vessel_id,
                name=probe_name,
                coordinate=float(position),
            )
        )

    def solve(
        self,
        *,
        t_end: float,
        t0: float = 0.0,
        record_every: int | None = None,
        snapshot_sample_points_by_vessel: dict[str, np.ndarray] | None = None,
        show_progress: bool = True,
        progress_description: str = "Solving network",
    ) -> Results:
        """Build the configured solver network and run the simulation."""

        boundaries = self._build_boundary_conditions()
        network = build_vascular_network(
            config=self.config,
            solver=self.solver_settings,
            external_boundaries=boundaries,
        )
        network.require_complete()

        solver = NetworkSolver(network)
        raw_result = solver.run(
            config=TimeConfig(
                t0=t0,
                t_end=t_end,
                fixed_dt=self.solver_settings.dt,
                cfl=self.solver_settings.cfl,
                max_steps=self.solver_settings.max_steps,
            ),
            record_every=(
                self.solver_settings.record_every
                if record_every is None
                else record_every
            ),
            probes=list(self._probes),
            snapshot_sample_points_by_vessel=snapshot_sample_points_by_vessel,
            show_progress=show_progress,
            progress_description=progress_description,
        )

        result = Results(
            raw=raw_result,
            solver_settings=self.solver_settings,
            metadata={
                "source_path": (
                    str(self.config.source_path)
                    if self.config.source_path is not None
                    else None
                )
            },
        )
        self._last_result = result
        return result

    def convergence_test(
        self,
        *,
        h_levels: list[float],
        dt_levels: list[float],
        expected_order: float,
        t_end: float = 1.0,
        show_progress: bool = False,
    ) -> "ConvergenceStudy":
        """Run a simple high-level convergence study using repeated solves."""

        if len(h_levels) != len(dt_levels):
            raise ValueError("h_levels and dt_levels must have the same length.")
        if len(h_levels) < 2:
            raise ValueError("At least two levels are required.")

        refinement_ratio = _validate_refinement_ratio(h_levels)
        snapshot_points = _snapshot_sample_points_by_vessel(
            config=self.config,
            h=h_levels[-1],
        )

        levels: list[ConvergenceStudyLevel] = []
        original_settings = self.solver_settings

        try:
            for index, (h, dt) in enumerate(zip(h_levels, dt_levels)):
                level_name = f"L{index}"
                progress_description = _convergence_progress_description(
                    level_name=level_name,
                    level_index=index,
                    total_levels=len(h_levels),
                    settings=original_settings,
                    h=h,
                    dt=dt,
                    t_end=t_end,
                )
                if show_progress:
                    print(progress_description, flush=True)

                self.solver_settings = replace(original_settings, h=h, dt=dt)
                result = self.solve(
                    t_end=t_end,
                    snapshot_sample_points_by_vessel=snapshot_points,
                    show_progress=show_progress,
                )
                levels.append(
                    ConvergenceStudyLevel(
                        name=level_name,
                        h=float(h),
                        dt=float(dt),
                        num_cells=_network_num_cells(result),
                        result=result,
                    )
                )
        finally:
            self.solver_settings = original_settings

        return ConvergenceStudy(
            levels=levels,
            expected_order=expected_order,
            refinement_ratio=refinement_ratio,
        )

    def _install_default_boundaries_from_config(self) -> None:
        for endpoint in self.config.external_endpoints():
            label = _normalize_boundary_kind(self.config.endpoint_label(endpoint) or "")
            if label in {"outflow", "outlet", "nonreflecting", "non-reflecting"}:
                self._boundaries[endpoint] = BoundaryAssignment(
                    endpoint=endpoint,
                    kind="nonreflecting",
                )

    def _resolve_external_endpoint(
        self,
        vessel_id: str,
        side: EndpointSide | str | None,
        *,
        role: str,
    ) -> NetworkEndpoint:
        self.config.vessel(vessel_id)

        if side is not None:
            endpoint = NetworkEndpoint(vessel_id, parse_endpoint_side(side))
            if endpoint not in self.config.external_endpoints():
                raise ValueError(f"Endpoint {endpoint.label()} is not an external boundary.")
            return endpoint

        candidates = [
            endpoint
            for endpoint in sorted(
                self.config.external_endpoints(),
                key=lambda item: item.label(),
            )
            if endpoint.vessel_id == vessel_id
        ]

        role_labels = _role_labels(role)
        labelled = [
            endpoint
            for endpoint in candidates
            if _normalize_boundary_kind(self.config.endpoint_label(endpoint) or "")
            in role_labels
        ]

        if len(labelled) == 1:
            return labelled[0]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Vessel {vessel_id!r} has no external endpoints.")

        labels = ", ".join(endpoint.label() for endpoint in candidates)
        raise ValueError(
            f"Could not infer {role} side for vessel {vessel_id!r}; choose one of: {labels}."
        )

    def _build_boundary_conditions(self) -> dict[NetworkEndpoint, BoundaryCondition]:
        boundaries: dict[NetworkEndpoint, BoundaryCondition] = {}

        missing = self.config.external_endpoints() - set(self._boundaries)
        if missing:
            labels = sorted(endpoint.label() for endpoint in missing)
            raise ValueError(f"Missing external boundary conditions for: {labels}.")

        for endpoint, assignment in self._boundaries.items():
            boundaries[endpoint] = self._make_boundary_condition(assignment)

        return boundaries

    def _make_boundary_condition(
        self,
        assignment: BoundaryAssignment,
    ) -> BoundaryCondition:
        kind = _normalize_boundary_kind(assignment.kind)
        vessel = self.config.vessel(assignment.endpoint.vessel_id)

        if kind == "nonreflecting":
            return NonReflectingBoundary()

        if assignment.function is None:
            raise ValueError(f"Boundary {assignment.endpoint.label()} requires a function.")

        if kind in {"flow", "flowrate", "flow_rate", "q"}:
            return PrescribedFlowBoundary(assignment.function)

        if kind in {"velocity", "v"}:
            return PrescribedFlowBoundary(
                lambda t, fn=assignment.function, area0=vessel.area0: area0 * fn(t)
            )

        if kind in {"area", "a"}:
            return PrescribedAreaBoundary(assignment.function)

        if kind in {"pressure", "p"}:
            return PrescribedPressureBoundary(assignment.function)

        raise ValueError(f"Unsupported boundary kind {assignment.kind!r}.")


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


def load_from_config(path: str | Path) -> HemodynamicModel:
    """Load a high-level hemodynamic model from a JSON file."""

    return HemodynamicModel.from_config(path)


NetworkModel = HemodynamicModel


def _normalize_boundary_kind(kind: str) -> str:
    return str(kind).strip().lower().replace("-", "_").replace(" ", "_")


def _role_labels(role: str) -> set[str]:
    if role == "inlet":
        return {"inflow", "inlet", "velocity", "flow", "flow_rate"}
    if role == "outlet":
        return {"outflow", "outlet", "nonreflecting", "non_reflecting"}
    return set()


def _validate_refinement_ratio(h_levels: list[float]) -> float:
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


def _snapshot_sample_points_by_vessel(
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


def _network_num_cells(result: Results) -> int:
    total = 0
    for vessel in result.network.vessels.values():
        num_cells = getattr(vessel.discretization, "num_cells", None)
        if num_cells is None:
            raise ValueError(
                f"Cannot determine num_cells for vessel {vessel.vessel_id!r}."
            )
        total += int(num_cells)
    return total


def _convergence_progress_description(
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
    "BoundaryAssignment",
    "ConvergenceStudy",
    "ConvergenceStudyLevel",
    "HemodynamicModel",
    "NetworkModel",
    "load_from_config",
    "read_area_csv",
    "read_flow_rate_csv",
    "read_velocity_csv",
]
