from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from hemo1d.lumped import CapillaryBedEndpoint, LumpedCapillaryBed
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
        self._lumped_beds: list[LumpedCapillaryBed] = []
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
        self._ensure_endpoint_not_lumped(endpoint)
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
        self._ensure_endpoint_not_lumped(endpoint)
        self._boundaries[endpoint] = BoundaryAssignment(
            endpoint=endpoint,
            kind=normalize_boundary_kind(kind),
            function=function,
        )

    def set_windkessel_outlet(
        self,
        *,
        vessel_id: str,
        R_art: float,
        C: float,
        R_ven: float,
        P_ven: float,
        P0: float | None = None,
        tissue_volume: float | None = None,
        side: EndpointSide | str | None = None,
        bed_id: str | None = None,
    ) -> None:
        """Connect one terminal outlet to a one-endpoint lumped capillary bed."""

        endpoint = self._resolve_external_endpoint(vessel_id, side, role="outlet")
        bed = LumpedCapillaryBed(
            bed_id=bed_id or f"{endpoint.vessel_id}_{endpoint.side.value}_windkessel",
            endpoints=[
                CapillaryBedEndpoint(
                    endpoint=endpoint,
                    resistance=float(R_art),
                )
            ],
            compliance=float(C),
            venous_resistance=float(R_ven),
            venous_pressure=float(P_ven),
            pressure=(
                self._default_bed_pressure([endpoint]) if P0 is None else float(P0)
            ),
            tissue_volume=tissue_volume,
        )

        self._add_lumped_bed(bed)

    def add_capillary_bed(
        self,
        *,
        bed_id: str,
        outlets: list[dict[str, Any] | tuple[Any, ...]],
        C: float,
        R_ven: float,
        P_ven: float,
        P0: float | None = None,
        tissue_volume: float | None = None,
    ) -> None:
        """Connect one or more terminal outlets to a shared capillary bed."""

        if not outlets:
            raise ValueError("Capillary bed outlets must be non-empty.")

        bed_endpoints = [self._parse_capillary_bed_outlet(outlet) for outlet in outlets]
        endpoints = [bed_endpoint.endpoint for bed_endpoint in bed_endpoints]

        bed = LumpedCapillaryBed(
            bed_id=bed_id,
            endpoints=bed_endpoints,
            compliance=float(C),
            venous_resistance=float(R_ven),
            venous_pressure=float(P_ven),
            pressure=(
                self._default_bed_pressure(endpoints) if P0 is None else float(P0)
            ),
            tissue_volume=tissue_volume,
        )

        self._add_lumped_bed(bed)

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
            lumped_beds=copy.deepcopy(self._lumped_beds),
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

        lumped_endpoints = self._lumped_endpoint_set()
        missing = self.config.external_endpoints() - set(self._boundaries) - lumped_endpoints
        if missing:
            labels = sorted(endpoint.label() for endpoint in missing)
            raise ValueError(f"Missing external boundary conditions for: {labels}.")

        for endpoint, assignment in self._boundaries.items():
            if endpoint in lumped_endpoints:
                raise ValueError(
                    f"Endpoint {endpoint.label()} cannot have both an ordinary "
                    "boundary condition and a lumped capillary bed."
                )
            boundaries[endpoint] = self._make_boundary_condition(assignment)

        return boundaries

    def _make_boundary_condition(
        self,
        assignment: BoundaryAssignment,
    ) -> BoundaryCondition:
        return make_boundary_condition(self.config, assignment)

    def _parse_capillary_bed_outlet(
        self,
        outlet: dict[str, Any] | tuple[Any, ...],
    ) -> CapillaryBedEndpoint:
        if isinstance(outlet, dict):
            vessel_id = outlet.get("vessel_id", outlet.get("vessel", outlet.get("id")))
            if vessel_id is None:
                raise ValueError("Capillary bed outlet is missing vessel_id.")
            if "R_art" not in outlet:
                raise ValueError("Capillary bed outlet is missing R_art.")
            side = outlet.get("side")
            resistance = outlet["R_art"]
        elif isinstance(outlet, tuple):
            if len(outlet) not in (2, 3):
                raise ValueError(
                    "Capillary bed outlet tuples must be "
                    "(vessel_id, R_art) or (vessel_id, R_art, side)."
                )
            vessel_id = outlet[0]
            resistance = outlet[1]
            side = outlet[2] if len(outlet) == 3 else None
        else:
            raise ValueError("Capillary bed outlets must be dicts or tuples.")

        endpoint = self._resolve_external_endpoint(str(vessel_id), side, role="outlet")
        return CapillaryBedEndpoint(
            endpoint=endpoint,
            resistance=float(resistance),
        )

    def _add_lumped_bed(self, bed: LumpedCapillaryBed) -> None:
        if any(existing.bed_id == bed.bed_id for existing in self._lumped_beds):
            raise ValueError(f"Lumped capillary bed id {bed.bed_id!r} already exists.")

        existing_endpoints = self._lumped_endpoint_set()
        overlap = existing_endpoints & bed.endpoint_set()
        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise ValueError(f"Endpoints already assigned to a lumped bed: {labels}.")

        for endpoint in bed.endpoint_set():
            self._boundaries.pop(endpoint, None)

        self._lumped_beds.append(bed)

    def _lumped_endpoint_set(self) -> set[NetworkEndpoint]:
        endpoints: set[NetworkEndpoint] = set()
        for bed in self._lumped_beds:
            endpoints.update(bed.endpoint_set())
        return endpoints

    def _ensure_endpoint_not_lumped(self, endpoint: NetworkEndpoint) -> None:
        if endpoint in self._lumped_endpoint_set():
            raise ValueError(
                f"Endpoint {endpoint.label()} is already assigned to a lumped capillary bed."
            )

    def _default_bed_pressure(self, endpoints: list[NetworkEndpoint]) -> float:
        if not endpoints:
            raise ValueError("Cannot infer capillary bed pressure without endpoints.")

        pressures = []
        for endpoint in endpoints:
            vessel = self.config.vessel(endpoint.vessel_id)
            pressures.append(float(vessel.p_ext + vessel.p0))

        return float(np.mean(pressures))


def load_from_config(path: str | Path) -> HemodynamicModel:
    """Load a high-level hemodynamic model from a JSON file."""

    return HemodynamicModel.from_config(path)


NetworkModel = HemodynamicModel


__all__ = [
    "HemodynamicModel",
    "NetworkModel",
    "load_from_config",
]
