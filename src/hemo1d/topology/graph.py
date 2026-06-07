from __future__ import annotations

from dataclasses import dataclass, field

from hemo1d.boundary import BoundaryCondition
from hemo1d.core.state import EndpointSide
from hemo1d.lumped import LumpedCapillaryBed
from hemo1d.topology.endpoint import NetworkEndpoint
from hemo1d.solvers.vessel import Vessel


@dataclass(frozen=True)
class Junction:
    """
    A two- or three-vessel junction.

    Endpoints are ordered. Endpoint 0 is the pressure reference used by the
    junction residual.
    """

    endpoints: tuple[NetworkEndpoint, ...]
    angles: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if len(endpoints) not in (2, 3):
            raise ValueError("Junction must connect exactly 2 or 3 endpoints.")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("Junction endpoints must be distinct.")

        angles = (None,) * len(endpoints) if self.angles is None else tuple(self.angles)
        if len(angles) != len(endpoints):
            raise ValueError("Junction angles must match the number of endpoints.")

        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "angles", angles)


@dataclass
class VascularNetwork:
    """
    General vascular network graph.

    A single-vessel problem is represented as:

        1 vessel
        0 junctions
        2 external boundaries

    A three-vessel junction is represented as:

        3 vessels
        1 junction
        3 external boundaries

    A larger arterial tree is represented as:

        many vessels
        many junctions
        many external boundaries
    """

    vessels: dict[str, Vessel]
    external_boundaries: dict[NetworkEndpoint, BoundaryCondition] = field(default_factory=dict)
    junctions: list[Junction] = field(default_factory=list)
    lumped_beds: list[LumpedCapillaryBed] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._validate_vessel_ids()
        self._validate_junction_endpoints()
        self._validate_external_boundary_endpoints()
        self._validate_lumped_bed_endpoints()
        self._validate_no_duplicate_junction_endpoints()
        self._validate_no_duplicate_lumped_bed_ids()
        self._validate_no_duplicate_lumped_bed_endpoints()
        self._validate_no_endpoint_is_both_boundary_and_junction()
        self._validate_no_endpoint_is_both_boundary_and_lumped_bed()
        self._validate_no_endpoint_is_both_junction_and_lumped_bed()

    def _validate_vessel_ids(self) -> None:
        for vessel_id, vessel in self.vessels.items():
            if vessel_id != vessel.vessel_id:
                raise ValueError(
                    f"Vessel dictionary key {vessel_id!r} does not match "
                    f"vessel.vessel_id {vessel.vessel_id!r}."
                )

    def _validate_endpoint_exists(self, endpoint: NetworkEndpoint) -> None:
        if endpoint.vessel_id not in self.vessels:
            raise ValueError(
                f"Endpoint {endpoint.label()} refers to unknown vessel {endpoint.vessel_id!r}."
            )

        if endpoint.side not in (EndpointSide.LEFT, EndpointSide.RIGHT):
            raise ValueError(f"Invalid endpoint side for {endpoint!r}.")

    def _validate_junction_endpoints(self) -> None:
        for junction in self.junctions:
            for endpoint in junction.endpoints:
                self._validate_endpoint_exists(endpoint)

    def _validate_external_boundary_endpoints(self) -> None:
        for endpoint in self.external_boundaries:
            self._validate_endpoint_exists(endpoint)

    def _validate_lumped_bed_endpoints(self) -> None:
        for bed in self.lumped_beds:
            for endpoint in bed.endpoint_set():
                self._validate_endpoint_exists(endpoint)

    def _junction_endpoints(self) -> list[NetworkEndpoint]:
        endpoints: list[NetworkEndpoint] = []

        for junction in self.junctions:
            endpoints.extend(junction.endpoints)

        return endpoints

    def _lumped_bed_endpoints(self) -> list[NetworkEndpoint]:
        endpoints: list[NetworkEndpoint] = []

        for bed in self.lumped_beds:
            endpoints.extend(bed.endpoint_set())

        return endpoints

    def _validate_no_duplicate_junction_endpoints(self) -> None:
        endpoints = self._junction_endpoints()

        if len(endpoints) != len(set(endpoints)):
            raise ValueError("At least one endpoint appears in more than one junction.")

    def _validate_no_duplicate_lumped_bed_ids(self) -> None:
        bed_ids = [bed.bed_id for bed in self.lumped_beds]

        if len(bed_ids) != len(set(bed_ids)):
            raise ValueError("Lumped capillary bed ids must be distinct.")

    def _validate_no_duplicate_lumped_bed_endpoints(self) -> None:
        endpoints = self._lumped_bed_endpoints()

        if len(endpoints) != len(set(endpoints)):
            raise ValueError("At least one endpoint appears in more than one lumped bed.")

    def _validate_no_endpoint_is_both_boundary_and_junction(self) -> None:
        junction_endpoints = set(self._junction_endpoints())
        boundary_endpoints = set(self.external_boundaries)

        overlap = junction_endpoints & boundary_endpoints

        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise ValueError(
                f"Endpoints cannot be both external boundaries and junction endpoints: {labels}."
            )

    def _validate_no_endpoint_is_both_boundary_and_lumped_bed(self) -> None:
        bed_endpoints = set(self._lumped_bed_endpoints())
        boundary_endpoints = set(self.external_boundaries)

        overlap = bed_endpoints & boundary_endpoints

        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise ValueError(
                "Endpoints cannot be both external boundaries and lumped bed "
                f"endpoints: {labels}."
            )

    def _validate_no_endpoint_is_both_junction_and_lumped_bed(self) -> None:
        bed_endpoints = set(self._lumped_bed_endpoints())
        junction_endpoints = set(self._junction_endpoints())

        overlap = bed_endpoints & junction_endpoints

        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise ValueError(
                f"Endpoints cannot be both junction and lumped bed endpoints: {labels}."
            )

    def endpoint_state_map_keys(self) -> set[NetworkEndpoint]:
        """
        Return all endpoints that must receive a BoundaryState during a time step.
        """
        keys = set(self.external_boundaries)

        for junction in self.junctions:
            keys.update(junction.endpoints)

        for bed in self.lumped_beds:
            keys.update(bed.endpoint_set())

        return keys

    def vessel_ids(self) -> list[str]:
        return sorted(self.vessels)

    def endpoints_for_vessel(self, vessel_id: str) -> tuple[NetworkEndpoint, NetworkEndpoint]:
        if vessel_id not in self.vessels:
            raise KeyError(f"Unknown vessel id {vessel_id!r}.")

        return (
            NetworkEndpoint(vessel_id, EndpointSide.LEFT),
            NetworkEndpoint(vessel_id, EndpointSide.RIGHT),
        )

    def all_vessel_endpoints(self) -> set[NetworkEndpoint]:
        endpoints: set[NetworkEndpoint] = set()

        for vessel_id in self.vessels:
            left, right = self.endpoints_for_vessel(vessel_id)
            endpoints.add(left)
            endpoints.add(right)

        return endpoints

    def unassigned_endpoints(self) -> set[NetworkEndpoint]:
        """
        Return endpoints not assigned to a boundary, junction, or lumped bed.
        """
        return self.all_vessel_endpoints() - self.endpoint_state_map_keys()

    def is_complete(self) -> bool:
        """
        Whether every vessel endpoint is assigned to a boundary, junction, or lumped bed.
        """
        return len(self.unassigned_endpoints()) == 0

    def require_complete(self) -> None:
        """
        Raise if any endpoint is not assigned.
        """
        unassigned = self.unassigned_endpoints()

        if unassigned:
            labels = sorted(endpoint.label() for endpoint in unassigned)
            raise ValueError(f"Network has unassigned endpoints: {labels}.")
