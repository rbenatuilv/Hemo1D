from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from hemo1d.api.boundaries import (
    BoundaryAssignment,
    ScalarFunction,
    make_boundary_condition,
    normalize_boundary_kind,
    role_labels,
)
from hemo1d.api.convergence import (
    ConvergenceStudy,
    ConvergenceStudyLevel,
    convergence_progress_description,
    network_num_cells,
    snapshot_sample_points_by_vessel,
    validate_refinement_ratio,
)
from hemo1d.boundary import BoundaryCondition
from hemo1d.builder import SolverSettings, build_vascular_network
from hemo1d.config import NetworkConfig, load_network_config, parse_endpoint_side
from hemo1d.core.state import EndpointSide
from hemo1d.observe import ProbePoint
from hemo1d.results import Results
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.time import TimeConfig
from hemo1d.topology.endpoint import NetworkEndpoint


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
            kind=normalize_boundary_kind(kind),
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
            kind=normalize_boundary_kind(kind),
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
    ) -> ConvergenceStudy:
        """Run a simple high-level convergence study using repeated solves."""

        if len(h_levels) != len(dt_levels):
            raise ValueError("h_levels and dt_levels must have the same length.")
        if len(h_levels) < 2:
            raise ValueError("At least two levels are required.")

        refinement_ratio = validate_refinement_ratio(h_levels)
        snapshot_points = snapshot_sample_points_by_vessel(
            config=self.config,
            h=h_levels[-1],
        )

        levels: list[ConvergenceStudyLevel] = []
        original_settings = self.solver_settings

        try:
            for index, (h, dt) in enumerate(zip(h_levels, dt_levels)):
                level_name = f"L{index}"
                progress_description = convergence_progress_description(
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
                        num_cells=network_num_cells(result),
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
            label = normalize_boundary_kind(self.config.endpoint_label(endpoint) or "")
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

        labelled = [
            endpoint
            for endpoint in candidates
            if normalize_boundary_kind(self.config.endpoint_label(endpoint) or "")
            in role_labels(role)
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
        return make_boundary_condition(self.config, assignment)


def load_from_config(path: str | Path) -> HemodynamicModel:
    """Load a high-level hemodynamic model from a JSON file."""

    return HemodynamicModel.from_config(path)


NetworkModel = HemodynamicModel


__all__ = [
    "HemodynamicModel",
    "NetworkModel",
    "load_from_config",
]
