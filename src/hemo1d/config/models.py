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
    """Declarative two- or three-vessel junction topology."""

    junction_id: str
    endpoints: tuple[NetworkEndpoint, ...]
    angles: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if len(endpoints) not in (2, 3):
            raise ValueError("JunctionConfig must contain exactly 2 or 3 endpoints.")

        angles = (None,) * len(endpoints) if self.angles is None else tuple(self.angles)
        if len(angles) != len(endpoints):
            raise ValueError("JunctionConfig angles must match the number of endpoints.")

        object.__setattr__(self, "junction_id", str(self.junction_id))
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "angles", angles)


@dataclass(frozen=True)
class CapillaryBedOutletConfig:
    """Declarative terminal endpoint feeding a lumped capillary bed."""

    vessel_id: str
    resistance: float
    side: EndpointSide | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vessel_id", str(self.vessel_id))
        object.__setattr__(self, "resistance", float(self.resistance))


@dataclass(frozen=True)
class CapillaryBedConfig:
    """Declarative lumped capillary bed coupled to one or more outlets."""

    bed_id: str
    outlets: tuple[CapillaryBedOutletConfig, ...]
    compliance: float
    venous_resistance: float
    venous_pressure: float
    initial_pressure: float | None = None
    tissue_volume: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bed_id", str(self.bed_id))
        object.__setattr__(self, "outlets", tuple(self.outlets))
        object.__setattr__(self, "compliance", float(self.compliance))
        object.__setattr__(self, "venous_resistance", float(self.venous_resistance))
        object.__setattr__(self, "venous_pressure", float(self.venous_pressure))
        if self.initial_pressure is not None:
            object.__setattr__(self, "initial_pressure", float(self.initial_pressure))
        if self.tissue_volume is not None:
            object.__setattr__(self, "tissue_volume", float(self.tissue_volume))


@dataclass(frozen=True)
class NetworkConfig:
    """Validated network configuration independent of any solver state."""

    vessels: dict[str, VesselConfig]
    junctions: list[JunctionConfig] = field(default_factory=list)
    source_path: Path | None = None
    capillary_beds: list[CapillaryBedConfig] = field(default_factory=list)

    def vessel(self, vessel_id: str) -> VesselConfig:
        try:
            return self.vessels[vessel_id]
        except KeyError as exc:
            raise KeyError(f"Unknown vessel id {vessel_id!r}.") from exc

    def junction_endpoints(self) -> set[NetworkEndpoint]:
        endpoints: set[NetworkEndpoint] = set()
        for junction in self.junctions:
            endpoints.update(junction.endpoints)
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
    "CapillaryBedConfig",
    "CapillaryBedOutletConfig",
    "JunctionConfig",
    "NetworkConfig",
    "VesselConfig",
]
