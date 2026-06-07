from __future__ import annotations

from hemo1d.config.models import NetworkConfig
from hemo1d.topology.endpoint import NetworkEndpoint


def validate_network_config(config: NetworkConfig) -> None:
    """Raise a useful error if the declarative network is inconsistent."""

    if not config.vessels:
        raise ValueError("Network config must contain at least one vessel.")

    for vessel_id, vessel in config.vessels.items():
        if vessel_id != vessel.vessel_id:
            raise ValueError(
                f"Vessel key {vessel_id!r} does not match vessel id {vessel.vessel_id!r}."
            )
        if vessel.length <= 0.0:
            raise ValueError(f"Vessel {vessel_id!r} length must be positive.")
        if vessel.area0 <= 0.0:
            raise ValueError(f"Vessel {vessel_id!r} area0 must be positive.")
        if vessel.beta <= 0.0:
            raise ValueError(f"Vessel {vessel_id!r} beta must be positive.")
        if vessel.blood.rho <= 0.0:
            raise ValueError(f"Vessel {vessel_id!r} blood density must be positive.")
        if vessel.blood.mu < 0.0:
            raise ValueError(f"Vessel {vessel_id!r} blood viscosity must be non-negative.")

    seen: set[NetworkEndpoint] = set()
    for junction in config.junctions:
        endpoints = junction.endpoints
        if len(endpoints) not in (2, 3):
            raise ValueError(f"Junction {junction.junction_id!r} must use 2 or 3 endpoints.")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(f"Junction {junction.junction_id!r} must use distinct endpoints.")
        if len(junction.angles) != len(endpoints):
            raise ValueError(f"Junction {junction.junction_id!r} angles must match endpoints.")
        for endpoint in endpoints:
            if endpoint.vessel_id not in config.vessels:
                raise ValueError(
                    f"Junction {junction.junction_id!r} references unknown "
                    f"vessel {endpoint.vessel_id!r}."
                )
            if endpoint in seen:
                raise ValueError(f"Endpoint {endpoint.label()} appears in more than one junction.")
            seen.add(endpoint)

    seen_beds: set[str] = set()
    for bed in config.capillary_beds:
        if not bed.bed_id.strip():
            raise ValueError("Capillary bed id must be non-empty.")
        if bed.bed_id in seen_beds:
            raise ValueError(f"Duplicate capillary bed id {bed.bed_id!r}.")
        seen_beds.add(bed.bed_id)

        if not bed.outlets:
            raise ValueError(f"Capillary bed {bed.bed_id!r} outlets must be non-empty.")
        if bed.compliance <= 0.0:
            raise ValueError(f"Capillary bed {bed.bed_id!r} compliance must be positive.")
        if bed.venous_resistance <= 0.0:
            raise ValueError(
                f"Capillary bed {bed.bed_id!r} venous resistance must be positive."
            )
        if bed.tissue_volume is not None and bed.tissue_volume <= 0.0:
            raise ValueError(
                f"Capillary bed {bed.bed_id!r} tissue volume must be positive."
            )

        for outlet in bed.outlets:
            if outlet.vessel_id not in config.vessels:
                raise ValueError(
                    f"Capillary bed {bed.bed_id!r} references unknown "
                    f"vessel {outlet.vessel_id!r}."
                )
            if outlet.resistance <= 0.0:
                raise ValueError(
                    f"Capillary bed {bed.bed_id!r} outlet resistance must be positive."
                )


__all__ = ["validate_network_config"]
