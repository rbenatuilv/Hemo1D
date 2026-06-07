from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hemo1d.core.state import BoundaryState
from hemo1d.lumped._capillary_bed_data import prepare_endpoint_data
from hemo1d.lumped._capillary_bed_equations import (
    endpoint_states_from_solution,
    initial_guess,
    residual as capillary_bed_residual,
    venous_outflow,
)
from hemo1d.lumped._capillary_bed_newton import newton_scales, solve_newton
from hemo1d.topology.endpoint import NetworkEndpoint

if TYPE_CHECKING:
    from hemo1d.solvers.vessel import Vessel


@dataclass(frozen=True)
class CapillaryBedEndpoint:
    """
    One 1D terminal endpoint feeding a lumped capillary bed.

    resistance is the terminal/arteriolar resistance between the 1D terminal
    pressure and the shared bed pressure.
    """

    endpoint: NetworkEndpoint
    resistance: float

    def __post_init__(self) -> None:
        if self.resistance <= 0.0:
            raise ValueError("Capillary bed endpoint resistance must be positive.")


@dataclass(frozen=True)
class CapillaryBedSample:
    """One diagnostic sample from a lumped capillary bed."""

    time: float
    bed_id: str
    pressure: float
    total_inflow: float
    venous_outflow: float
    regional_perfusion: float | None
    endpoint_inflows: dict[NetworkEndpoint, float]


@dataclass
class LumpedCapillaryBed:
    """
    Shared 0D capillary bed coupled to one or more 1D terminal endpoints.

    The solve is implicit in the bed pressure and in each boundary area. Each
    boundary flow is eliminated through the outgoing characteristic equation.
    """

    bed_id: str
    endpoints: list[CapillaryBedEndpoint]
    compliance: float
    venous_resistance: float
    venous_pressure: float
    pressure: float
    tissue_volume: float | None = None
    last_total_inflow: float = 0.0
    last_venous_outflow: float = 0.0
    last_endpoint_inflows: dict[NetworkEndpoint, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bed_id:
            raise ValueError("Capillary bed id must be non-empty.")
        if not self.endpoints:
            raise ValueError("Capillary bed must connect at least one endpoint.")
        if len({bed_endpoint.endpoint for bed_endpoint in self.endpoints}) != len(
            self.endpoints
        ):
            raise ValueError(f"Capillary bed {self.bed_id!r} has duplicate endpoints.")
        if self.compliance <= 0.0:
            raise ValueError("Capillary bed compliance must be positive.")
        if self.venous_resistance <= 0.0:
            raise ValueError("Capillary bed venous resistance must be positive.")
        if self.tissue_volume is not None and self.tissue_volume <= 0.0:
            raise ValueError("Capillary bed tissue volume must be positive when provided.")

        self.endpoints = list(self.endpoints)
        self.pressure = float(self.pressure)
        self.venous_pressure = float(self.venous_pressure)
        self.compliance = float(self.compliance)
        self.venous_resistance = float(self.venous_resistance)
        self.last_total_inflow = float(self.last_total_inflow)
        self.last_venous_outflow = float(self.last_venous_outflow)
        self.last_endpoint_inflows = dict(self.last_endpoint_inflows)

    def endpoint_set(self) -> set[NetworkEndpoint]:
        return {bed_endpoint.endpoint for bed_endpoint in self.endpoints}

    def solve(
        self,
        vessels: dict[str, Vessel],
        dt: float,
    ) -> dict[NetworkEndpoint, BoundaryState]:
        """
        Solve all coupled boundary states and update the bed pressure.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        data = prepare_endpoint_data(
            bed_id=self.bed_id,
            endpoints=self.endpoints,
            vessels=vessels,
            dt=dt,
        )
        pressure_old = self.pressure

        y0 = initial_guess(data, pressure_old)
        scales = newton_scales(
            data=data,
            pressure_old=pressure_old,
            venous_pressure=self.venous_pressure,
            venous_resistance=self.venous_resistance,
        )

        def residual(trial: np.ndarray) -> np.ndarray:
            return capillary_bed_residual(
                trial,
                data=data,
                compliance=self.compliance,
                venous_pressure=self.venous_pressure,
                venous_resistance=self.venous_resistance,
                pressure_old=pressure_old,
                dt=dt,
            )

        y = solve_newton(
            bed_id=self.bed_id,
            y0=y0,
            residual=residual,
            scales=scales,
        )
        endpoint_states, endpoint_inflows = endpoint_states_from_solution(y, data)
        self._update_state(
            pressure=float(y[-1]),
            endpoint_inflows=endpoint_inflows,
        )

        return endpoint_states

    def sample(self, time: float) -> CapillaryBedSample:
        endpoint_inflows = dict(self.last_endpoint_inflows)
        if not endpoint_inflows:
            endpoint_inflows = {
                bed_endpoint.endpoint: 0.0 for bed_endpoint in self.endpoints
            }

        total_inflow = float(self.last_total_inflow)
        bed_venous_outflow = self._venous_outflow(self.pressure)
        regional_perfusion = None
        if self.tissue_volume is not None:
            regional_perfusion = total_inflow / self.tissue_volume

        return CapillaryBedSample(
            time=float(time),
            bed_id=self.bed_id,
            pressure=float(self.pressure),
            total_inflow=total_inflow,
            venous_outflow=bed_venous_outflow,
            regional_perfusion=regional_perfusion,
            endpoint_inflows=endpoint_inflows,
        )

    def _update_state(
        self,
        *,
        pressure: float,
        endpoint_inflows: dict[NetworkEndpoint, float],
    ) -> None:
        self.pressure = float(pressure)
        self.last_endpoint_inflows = dict(endpoint_inflows)
        self.last_total_inflow = float(sum(endpoint_inflows.values()))
        self.last_venous_outflow = self._venous_outflow(pressure)

    def _venous_outflow(self, pressure: float) -> float:
        return venous_outflow(
            pressure,
            venous_pressure=self.venous_pressure,
            venous_resistance=self.venous_resistance,
        )


__all__ = [
    "CapillaryBedEndpoint",
    "CapillaryBedSample",
    "LumpedCapillaryBed",
]
