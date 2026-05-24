from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hemo1d.boundary.junction import compatibility_target, outgoing_left_eigenvector
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState
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

        data = self._prepare_endpoint_data(vessels=vessels, dt=dt)
        pressure_old = self.pressure

        y = np.empty(len(data) + 1, dtype=float)
        for i, item in enumerate(data):
            y[i] = item.area_n
        y[-1] = pressure_old

        def residual(trial: np.ndarray) -> np.ndarray:
            return self._residual(
                trial,
                data=data,
                pressure_old=pressure_old,
                dt=dt,
            )

        y = self._newton_solve(y0=y, residual=residual)

        pressure = float(y[-1])
        endpoint_states: dict[NetworkEndpoint, BoundaryState] = {}
        endpoint_inflows: dict[NetworkEndpoint, float] = {}
        total_inflow = 0.0

        for i, item in enumerate(data):
            area = float(y[i])
            flow_rate = item.flow_rate(area)
            inflow = item.endpoint.outward_flow(flow_rate)
            endpoint_states[item.endpoint] = BoundaryState(
                area=area,
                flow_rate=flow_rate,
            )
            endpoint_inflows[item.endpoint] = inflow
            total_inflow += inflow

        self.pressure = pressure
        self.last_endpoint_inflows = endpoint_inflows
        self.last_total_inflow = float(total_inflow)
        self.last_venous_outflow = float(
            (pressure - self.venous_pressure) / self.venous_resistance
        )

        return endpoint_states

    def sample(self, time: float) -> CapillaryBedSample:
        endpoint_inflows = dict(self.last_endpoint_inflows)
        if not endpoint_inflows:
            endpoint_inflows = {
                bed_endpoint.endpoint: 0.0 for bed_endpoint in self.endpoints
            }

        total_inflow = float(self.last_total_inflow)
        venous_outflow = float(
            (self.pressure - self.venous_pressure) / self.venous_resistance
        )
        regional_perfusion = None
        if self.tissue_volume is not None:
            regional_perfusion = total_inflow / self.tissue_volume

        return CapillaryBedSample(
            time=float(time),
            bed_id=self.bed_id,
            pressure=float(self.pressure),
            total_inflow=total_inflow,
            venous_outflow=venous_outflow,
            regional_perfusion=regional_perfusion,
            endpoint_inflows=endpoint_inflows,
        )

    def _prepare_endpoint_data(
        self,
        *,
        vessels: dict[str, Vessel],
        dt: float,
    ) -> list[_EndpointSolveData]:
        data: list[_EndpointSolveData] = []

        for bed_endpoint in self.endpoints:
            endpoint = bed_endpoint.endpoint
            try:
                vessel = vessels[endpoint.vessel_id]
            except KeyError as exc:
                raise KeyError(
                    f"Capillary bed {self.bed_id!r} endpoint {endpoint.label()} "
                    "refers to an unknown vessel."
                ) from exc

            endpoint_data = vessel.endpoint_data(endpoint.side)
            l_out = outgoing_left_eigenvector(
                vessel.physics,
                endpoint_data,
                endpoint.side,
            )
            l_area = float(l_out[0])
            l_flow = float(l_out[1])
            if abs(l_flow) < 1.0e-14:
                raise RuntimeError(
                    f"Cannot solve capillary bed {self.bed_id!r} at "
                    f"{endpoint.label()}: outgoing characteristic l_Q is too small."
                )

            target_vector = compatibility_target(vessel.physics, endpoint_data, dt)
            target = float(l_area * target_vector[0] + l_flow * target_vector[1])

            data.append(
                _EndpointSolveData(
                    endpoint=endpoint,
                    resistance=float(bed_endpoint.resistance),
                    physics=vessel.physics,
                    area_n=float(endpoint_data.state.area),
                    l_area=l_area,
                    l_flow=l_flow,
                    target=target,
                )
            )

        return data

    def _residual(
        self,
        y: np.ndarray,
        *,
        data: list[_EndpointSolveData],
        pressure_old: float,
        dt: float,
    ) -> np.ndarray:
        n = len(data)
        residual = np.empty(n + 1, dtype=float)

        if not np.all(np.isfinite(y)) or np.any(y[:n] <= 0.0):
            return np.full(n + 1, 1.0e30, dtype=float)

        pressure = float(y[-1])
        total_inflow = 0.0

        for i, item in enumerate(data):
            area = float(y[i])
            flow_rate = item.flow_rate(area)
            inflow = item.endpoint.outward_flow(flow_rate)
            pressure_1d = float(item.physics.pressure(area))

            residual[i] = inflow - (pressure_1d - pressure) / item.resistance
            total_inflow += inflow

        residual[-1] = (
            self.compliance * (pressure - pressure_old) / dt
            - total_inflow
            + (pressure - self.venous_pressure) / self.venous_resistance
        )

        if not np.all(np.isfinite(residual)):
            return np.full(n + 1, 1.0e30, dtype=float)

        return residual

    def _newton_solve(
        self,
        *,
        y0: np.ndarray,
        residual,
        max_iter: int = 30,
        tol_abs: float = 1.0e-10,
    ) -> np.ndarray:
        y = y0.astype(float, copy=True)
        r = residual(y)
        norm = _scaled_norm(r, y)

        if norm <= tol_abs:
            return y

        for _ in range(max_iter):
            jacobian = _finite_difference_jacobian(residual, y, r)
            try:
                delta = np.linalg.solve(jacobian, -r)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"Capillary bed {self.bed_id!r} Newton solve failed: "
                    "singular finite-difference Jacobian."
                ) from exc

            alpha = 1.0
            accepted = False
            best_y = y
            best_r = r
            best_norm = norm

            while alpha >= 1.0e-8:
                candidate = y + alpha * delta
                if np.any(candidate[:-1] <= 0.0):
                    alpha *= 0.5
                    continue

                candidate_r = residual(candidate)
                candidate_norm = _scaled_norm(candidate_r, candidate)
                if candidate_norm < norm:
                    best_y = candidate
                    best_r = candidate_r
                    best_norm = candidate_norm
                    accepted = True
                    break

                alpha *= 0.5

            if not accepted:
                raise RuntimeError(
                    f"Capillary bed {self.bed_id!r} Newton solve failed: "
                    f"residual did not decrease from {norm:.6e}."
                )

            y = best_y
            r = best_r
            norm = best_norm

            if norm <= tol_abs:
                return y

        raise RuntimeError(
            f"Capillary bed {self.bed_id!r} Newton solve did not converge; "
            f"final scaled residual {norm:.6e}."
        )


@dataclass(frozen=True)
class _EndpointSolveData:
    endpoint: NetworkEndpoint
    resistance: float
    physics: Hemo1DPhysics
    area_n: float
    l_area: float
    l_flow: float
    target: float

    def flow_rate(self, area: float) -> float:
        return float((self.target - self.l_area * area) / self.l_flow)


def _finite_difference_jacobian(residual, y: np.ndarray, r0: np.ndarray) -> np.ndarray:
    n = len(y)
    jacobian = np.empty((n, n), dtype=float)

    for j in range(n):
        eps = max(1.0e-8 * abs(float(y[j])), 1.0e-12)
        y_step = y.copy()
        y_step[j] += eps
        jacobian[:, j] = (residual(y_step) - r0) / eps

    return jacobian


def _scaled_norm(residual: np.ndarray, y: np.ndarray) -> float:
    scale = max(1.0, float(np.linalg.norm(y, ord=np.inf)))
    return float(np.linalg.norm(residual, ord=np.inf) / scale)


__all__ = [
    "CapillaryBedEndpoint",
    "CapillaryBedSample",
    "LumpedCapillaryBed",
]
