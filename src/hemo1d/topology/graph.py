from __future__ import annotations

from dataclasses import dataclass, field

from hemo1d.boundary import BoundaryCondition
from hemo1d.core.state import EndpointSide
from hemo1d.topology.endpoint import NetworkEndpoint
from hemo1d.solvers.vessel import Vessel


@dataclass(frozen=True)
class Bifurcation:
    """
    A 1-to-2 bifurcation.

    Each bifurcation connects exactly three vessel endpoints:

        parent endpoint
        daughter1 endpoint
        daughter2 endpoint

    The common orientation is:

        parent.RIGHT -> daughter1.LEFT + daughter2.LEFT

    but the endpoint sides are stored explicitly.
    """

    parent: NetworkEndpoint
    daughter1: NetworkEndpoint
    daughter2: NetworkEndpoint
    angles: tuple[float | None, float | None, float | None] = (None, None, None)

    def endpoints(self) -> tuple[NetworkEndpoint, NetworkEndpoint, NetworkEndpoint]:
        return self.parent, self.daughter1, self.daughter2


@dataclass
class VascularNetwork:
    """
    General vascular network graph.

    A single-vessel problem is represented as:

        1 vessel
        0 bifurcations
        2 external boundaries

    A three-vessel bifurcation is represented as:

        3 vessels
        1 bifurcation
        3 external boundaries

    A larger arterial tree is represented as:

        many vessels
        many bifurcations
        many external boundaries
    """

    vessels: dict[str, Vessel]
    bifurcations: list[Bifurcation] = field(default_factory=list)
    external_boundaries: dict[NetworkEndpoint, BoundaryCondition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._validate_vessel_ids()
        self._validate_bifurcation_endpoints()
        self._validate_external_boundary_endpoints()
        self._validate_no_duplicate_junction_endpoints()
        self._validate_no_endpoint_is_both_boundary_and_junction()

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
                f"Endpoint {endpoint.label()} refers to unknown vessel "
                f"{endpoint.vessel_id!r}."
            )

        if endpoint.side not in (EndpointSide.LEFT, EndpointSide.RIGHT):
            raise ValueError(f"Invalid endpoint side for {endpoint!r}.")

    def _validate_bifurcation_endpoints(self) -> None:
        for bifurcation in self.bifurcations:
            for endpoint in bifurcation.endpoints():
                self._validate_endpoint_exists(endpoint)

    def _validate_external_boundary_endpoints(self) -> None:
        for endpoint in self.external_boundaries:
            self._validate_endpoint_exists(endpoint)

    def _junction_endpoints(self) -> list[NetworkEndpoint]:
        endpoints: list[NetworkEndpoint] = []

        for bifurcation in self.bifurcations:
            endpoints.extend(bifurcation.endpoints())

        return endpoints

    def _validate_no_duplicate_junction_endpoints(self) -> None:
        endpoints = self._junction_endpoints()

        if len(endpoints) != len(set(endpoints)):
            raise ValueError(
                "At least one endpoint appears in more than one bifurcation."
            )

    def _validate_no_endpoint_is_both_boundary_and_junction(self) -> None:
        junction_endpoints = set(self._junction_endpoints())
        boundary_endpoints = set(self.external_boundaries)

        overlap = junction_endpoints & boundary_endpoints

        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise ValueError(
                "Endpoints cannot be both external boundaries and junction endpoints: "
                f"{labels}."
            )

    def endpoint_state_map_keys(self) -> set[NetworkEndpoint]:
        """
        Return all endpoints that must receive a BoundaryState during a time step.
        """
        keys = set(self.external_boundaries)

        for bifurcation in self.bifurcations:
            keys.update(bifurcation.endpoints())

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
        Return vessel endpoints not assigned to either an external boundary or a junction.
        """
        return self.all_vessel_endpoints() - self.endpoint_state_map_keys()

    def is_complete(self) -> bool:
        """
        Whether every vessel endpoint is assigned to a boundary or junction.
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