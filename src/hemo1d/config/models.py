from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hemo1d.core.state import EndpointSide
from hemo1d.topology.endpoint import NetworkEndpoint


@dataclass(frozen=True)
class BloodConfig:
    """Default blood parameters used when a vessel does not override them."""

    rho: float = 1.06
    mu: float = 0.035


@dataclass(frozen=True)
class VesselConfig:
    """Declarative parameters for one straight compliant vessel."""

    vessel_id: str
    length: float
    area0: float
    beta: float
    left_bound: str | None = None
    right_bound: str | None = None
    blood: BloodConfig = field(default_factory=BloodConfig)
    gamma_profile: float = 2.0
    p0: float = 0.0
    p_ext: float = 0.0
    gamma_pressure_loss: float = 0.0


@dataclass(frozen=True)
class JunctionConfig:
    """Declarative 1-to-2 bifurcation topology."""

    junction_id: str
    parent: NetworkEndpoint
    daughter1: NetworkEndpoint
    daughter2: NetworkEndpoint
    angles: tuple[float | None, float | None, float | None] = (None, None, None)

    def endpoints(self) -> tuple[NetworkEndpoint, NetworkEndpoint, NetworkEndpoint]:
        return self.parent, self.daughter1, self.daughter2


@dataclass(frozen=True)
class NetworkConfig:
    """Validated network configuration independent of any solver state."""

    vessels: dict[str, VesselConfig]
    bifurcations: list[JunctionConfig] = field(default_factory=list)
    source_path: Path | None = None

    def vessel(self, vessel_id: str) -> VesselConfig:
        try:
            return self.vessels[vessel_id]
        except KeyError as exc:
            raise KeyError(f"Unknown vessel id {vessel_id!r}.") from exc

    def junction_endpoints(self) -> set[NetworkEndpoint]:
        endpoints: set[NetworkEndpoint] = set()
        for bifurcation in self.bifurcations:
            endpoints.update(bifurcation.endpoints())
        return endpoints

    def all_endpoints(self) -> set[NetworkEndpoint]:
        endpoints: set[NetworkEndpoint] = set()
        for vessel_id in self.vessels:
            endpoints.add(NetworkEndpoint(vessel_id, EndpointSide.LEFT))
            endpoints.add(NetworkEndpoint(vessel_id, EndpointSide.RIGHT))
        return endpoints

    def external_endpoints(self) -> set[NetworkEndpoint]:
        return self.all_endpoints() - self.junction_endpoints()

    def endpoint_label(self, endpoint: NetworkEndpoint) -> str | None:
        vessel = self.vessel(endpoint.vessel_id)
        if endpoint.side == EndpointSide.LEFT:
            return vessel.left_bound
        if endpoint.side == EndpointSide.RIGHT:
            return vessel.right_bound
        raise ValueError(f"Unknown endpoint side: {endpoint.side!r}.")


__all__ = [
    "BloodConfig",
    "JunctionConfig",
    "NetworkConfig",
    "VesselConfig",
]
