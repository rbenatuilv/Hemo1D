from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hemo1d.config import NetworkConfig, VesselConfig
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.lumped import LumpedCapillaryBed
from hemo1d.topology.graph import Junction, VascularNetwork
from hemo1d.topology.endpoint import NetworkEndpoint
from hemo1d.boundary.base import BoundaryCondition


SolverMethod = Literal["cg", "dg"]


@dataclass(frozen=True)
class SolverSettings:
    """High-level spatial and temporal solver settings."""

    method: SolverMethod = "cg"
    h: float | None = None
    dt: float | None = None
    poly_order: int = 1
    num_cells: int | dict[str, int] | None = None
    cfl: float = float(np.sqrt(3.0) / 3.0)
    dg_time_scheme: Literal["euler", "rk2"] = "rk2"
    dg_flux: str = "lxf"
    record_every: int = 1
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        method = self.method.lower()
        if method not in {"cg", "dg"}:
            raise ValueError("Solver method must be 'CG' or 'DG'.")
        object.__setattr__(self, "method", method)

        if self.h is not None and self.h <= 0.0:
            raise ValueError("h must be positive when provided.")
        if self.dt is not None and self.dt <= 0.0:
            raise ValueError("dt must be positive when provided.")
        if self.poly_order <= 0:
            raise ValueError("poly_order must be positive.")
        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive.")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        from hemo1d.solvers.dg.flux import canonicalize_dg_flux_scheme

        object.__setattr__(
            self,
            "dg_flux",
            canonicalize_dg_flux_scheme(self.dg_flux),
        )


def build_vascular_network(
    *,
    config: NetworkConfig,
    solver: SolverSettings,
    external_boundaries: dict[NetworkEndpoint, BoundaryCondition],
    lumped_beds: list[LumpedCapillaryBed] | None = None,
) -> VascularNetwork:
    """Build a mutable solver network from declarative model state."""

    vessels = {
        vessel_id: _create_solver_vessel(
            vessel=vessel,
            solver=solver,
            num_cells=_num_cells_for_vessel(vessel, solver),
        )
        for vessel_id, vessel in config.vessels.items()
    }

    junctions = [
        Junction(
            endpoints=junction.endpoints,
            angles=junction.angles,
        )
        for junction in config.junctions
    ]

    return VascularNetwork(
        vessels=vessels,
        external_boundaries=external_boundaries,
        junctions=junctions,
        lumped_beds=list(lumped_beds or []),
    )


def make_physics(vessel: VesselConfig) -> Hemo1DPhysics:
    """Create the shared pointwise physics object for one vessel."""

    return Hemo1DPhysics(
        ModelParameters(
            blood=BloodParameters(rho=vessel.blood.rho, mu=vessel.blood.mu),
            vessel=VesselParameters(
                length=vessel.length,
                area0=vessel.area0,
                beta=vessel.beta,
            ),
            gamma_profile=vessel.gamma_profile,
            p0=vessel.p0,
            p_ext=vessel.p_ext,
            gamma_pressure_loss=vessel.gamma_pressure_loss,
        ),
        NP_BACKEND,
    )


def _create_solver_vessel(*, vessel: VesselConfig, solver: SolverSettings, num_cells: int):
    physics = make_physics(vessel)

    if solver.method == "cg":
        from hemo1d.solvers.cg.discretization import CGFEMDiscretization, CGMeshConfig
        from hemo1d.solvers.cg.factory import create_cg_vessel

        discretization = CGFEMDiscretization(
            CGMeshConfig(
                length=vessel.length,
                num_cells=num_cells,
                degree=solver.poly_order,
            )
        )
        solver_vessel = create_cg_vessel(
            vessel_id=vessel.vessel_id,
            physics=physics,
            discretization=discretization,
        )
    elif solver.method == "dg":
        from hemo1d.solvers.dg.discretization import DGFEMDiscretization, DGMeshConfig
        from hemo1d.solvers.dg.factory import create_dg_vessel

        discretization = DGFEMDiscretization(
            DGMeshConfig(
                length=vessel.length,
                num_cells=num_cells,
                degree=solver.poly_order,
            )
        )
        solver_vessel = create_dg_vessel(
            vessel_id=vessel.vessel_id,
            physics=physics,
            discretization=discretization,
            time_scheme=solver.dg_time_scheme,
            flux_scheme=solver.dg_flux,
        )
    else:
        raise RuntimeError(f"Unexpected solver method: {solver.method}")

    solver_vessel.interpolate_rest_state()
    return solver_vessel


def _num_cells_for_vessel(vessel: VesselConfig, solver: SolverSettings) -> int:
    if isinstance(solver.num_cells, dict):
        if vessel.vessel_id in solver.num_cells:
            return _validate_num_cells(solver.num_cells[vessel.vessel_id], vessel.vessel_id)
    elif isinstance(solver.num_cells, int):
        return _validate_num_cells(solver.num_cells, vessel.vessel_id)

    if solver.h is not None:
        return max(1, int(np.ceil(vessel.length / solver.h)))

    return 64


def _validate_num_cells(value: int, vessel_id: str) -> int:
    cells = int(value)
    if cells <= 0:
        raise ValueError(f"num_cells for vessel {vessel_id!r} must be positive.")
    return cells
