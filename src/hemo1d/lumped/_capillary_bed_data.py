from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from hemo1d.boundary.junction import compatibility_target, outgoing_left_eigenvector
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.topology.endpoint import NetworkEndpoint

if TYPE_CHECKING:
    from hemo1d.solvers.vessel import Vessel


class _CapillaryBedEndpointLike(Protocol):
    endpoint: NetworkEndpoint
    resistance: float


@dataclass(frozen=True)
class EndpointSolveData:
    endpoint: NetworkEndpoint
    resistance: float
    physics: Hemo1DPhysics
    area_n: float
    l_area: float
    l_flow: float
    target: float

    def flow_rate(self, area: float) -> float:
        return float((self.target - self.l_area * area) / self.l_flow)


def prepare_endpoint_data(
    *,
    bed_id: str,
    endpoints: Sequence[_CapillaryBedEndpointLike],
    vessels: dict[str, Vessel],
    dt: float,
) -> list[EndpointSolveData]:
    data: list[EndpointSolveData] = []

    for bed_endpoint in endpoints:
        endpoint = bed_endpoint.endpoint
        try:
            vessel = vessels[endpoint.vessel_id]
        except KeyError as exc:
            raise KeyError(
                f"Capillary bed {bed_id!r} endpoint {endpoint.label()} "
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
                f"Cannot solve capillary bed {bed_id!r} at "
                f"{endpoint.label()}: outgoing characteristic l_Q is too small."
            )

        target_vector = compatibility_target(vessel.physics, endpoint_data, dt)
        target = float(l_area * target_vector[0] + l_flow * target_vector[1])

        data.append(
            EndpointSolveData(
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
