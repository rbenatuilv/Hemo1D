from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def load_network_config(path: str | Path) -> NetworkConfig:
    """
    Load a network configuration from JSON.

    Supported forms:
        - a combined file with top-level ``vessels`` and optional
          ``bifurcations`` or ``junctions`` keys;
        - the existing project ``vessels.json`` shape, optionally paired with a
          sibling ``bifurcations.json`` file.
    """

    config_path = Path(path)
    data = _load_json_object(config_path)

    if "vessels" in data:
        vessel_data = data["vessels"]
        bifurcation_data = data.get("bifurcations", data.get("junctions", {}))
        defaults = data.get("defaults", {})
    else:
        vessel_data = data
        bifurcation_data = _load_sibling_bifurcations(config_path)
        defaults = {}

    vessels = _parse_vessels(vessel_data, defaults=defaults)
    bifurcations = _parse_bifurcations(bifurcation_data)

    config = NetworkConfig(
        vessels=vessels,
        bifurcations=bifurcations,
        source_path=config_path,
    )
    validate_network_config(config)
    return config


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
    for bifurcation in config.bifurcations:
        endpoints = bifurcation.endpoints()
        if len(set(endpoints)) != 3:
            raise ValueError(
                f"Bifurcation {bifurcation.junction_id!r} must use three distinct endpoints."
            )
        for endpoint in endpoints:
            if endpoint.vessel_id not in config.vessels:
                raise ValueError(
                    f"Bifurcation {bifurcation.junction_id!r} references unknown "
                    f"vessel {endpoint.vessel_id!r}."
                )
            if endpoint in seen:
                raise ValueError(
                    f"Endpoint {endpoint.label()} appears in more than one bifurcation."
                )
            seen.add(endpoint)


def parse_endpoint_side(value: EndpointSide | str) -> EndpointSide:
    """Parse endpoint side names used by config files and the public API."""

    if isinstance(value, EndpointSide):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"left", "l", "0", "z0"}:
        return EndpointSide.LEFT
    if normalized in {"right", "r", "1", "zl", "z_l"}:
        return EndpointSide.RIGHT

    raise ValueError(f"Invalid endpoint side {value!r}; expected 'left' or 'right'.")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top level in {path}.")

    return data


def _load_sibling_bifurcations(path: Path) -> Any:
    sibling = path.with_name("bifurcations.json")
    if sibling.exists():
        return _load_json_object(sibling)
    return {}


def _parse_vessels(data: Any, *, defaults: dict[str, Any]) -> dict[str, VesselConfig]:
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((_required_string(item, "id"), item) for item in data)
    else:
        raise ValueError("'vessels' must be a JSON object or list.")

    vessels: dict[str, VesselConfig] = {}
    default_blood = defaults.get("blood", {})

    for vessel_id, raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"Vessel {vessel_id!r} must be a JSON object.")
        if vessel_id in vessels:
            raise ValueError(f"Duplicate vessel id {vessel_id!r}.")

        blood_data = {
            **default_blood,
            **raw.get("blood", {}),
        }
        blood = BloodConfig(
            rho=float(blood_data.get("rho", defaults.get("rho", 1.06))),
            mu=float(blood_data.get("mu", defaults.get("mu", 0.035))),
        )

        vessels[str(vessel_id)] = VesselConfig(
            vessel_id=str(vessel_id),
            length=float(_pick(raw, "length")),
            area0=float(_pick(raw, "area0", "initial_area", "A0")),
            beta=float(_pick(raw, "beta", "beta_coeff")),
            left_bound=_optional_label(raw.get("left_bound", raw.get("left_boundary"))),
            right_bound=_optional_label(raw.get("right_bound", raw.get("right_boundary"))),
            blood=blood,
            gamma_profile=float(raw.get("gamma_profile", defaults.get("gamma_profile", 2.0))),
            p0=float(raw.get("p0", defaults.get("p0", 0.0))),
            p_ext=float(raw.get("p_ext", defaults.get("p_ext", 0.0))),
            gamma_pressure_loss=float(
                raw.get("gamma_pressure_loss", raw.get("gamma_pressure", defaults.get("gamma_pressure_loss", 0.0)))
            ),
        )

    return vessels


def _parse_bifurcations(data: Any) -> list[JunctionConfig]:
    if data in ({}, [], None):
        return []

    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((_required_string(item, "id"), item) for item in data)
    else:
        raise ValueError("'bifurcations'/'junctions' must be a JSON object or list.")

    bifurcations: list[JunctionConfig] = []
    for junction_id, raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"Bifurcation {junction_id!r} must be a JSON object.")

        if "branches" in raw:
            branches = raw["branches"]
            positions = raw["positions"]
            if len(branches) != 3 or len(positions) != 3:
                raise ValueError(
                    f"Bifurcation {junction_id!r} must have exactly three branches/positions."
                )
            parent = NetworkEndpoint(str(branches[0]), parse_endpoint_side(positions[0]))
            daughter1 = NetworkEndpoint(str(branches[1]), parse_endpoint_side(positions[1]))
            daughter2 = NetworkEndpoint(str(branches[2]), parse_endpoint_side(positions[2]))
            angles = _parse_angles(raw.get("angles"), junction_id=junction_id)
        else:
            parent = _parse_endpoint_ref(raw["parent"])
            daughters = raw.get("daughters", [raw.get("daughter1"), raw.get("daughter2")])
            if len(daughters) != 2:
                raise ValueError(f"Bifurcation {junction_id!r} must have two daughters.")
            daughter1 = _parse_endpoint_ref(daughters[0])
            daughter2 = _parse_endpoint_ref(daughters[1])
            angles = _parse_angles(raw.get("angles"), junction_id=junction_id)

        bifurcations.append(
            JunctionConfig(
                junction_id=str(junction_id),
                parent=parent,
                daughter1=daughter1,
                daughter2=daughter2,
                angles=angles,
            )
        )

    return bifurcations


def _parse_endpoint_ref(data: Any) -> NetworkEndpoint:
    if not isinstance(data, dict):
        raise ValueError("Endpoint reference must be an object with vessel_id and side.")

    vessel_id = data.get("vessel_id", data.get("vessel", data.get("id")))
    if vessel_id is None:
        raise ValueError("Endpoint reference is missing vessel_id.")

    return NetworkEndpoint(str(vessel_id), parse_endpoint_side(data.get("side")))


def _pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise ValueError(f"Missing required field; expected one of {keys}.")


def _required_string(data: dict[str, Any], key: str) -> str:
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"List entries must contain {key!r}.")
    return str(data[key])


def _optional_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    return label or None


def _parse_angles(value: Any, *, junction_id: str) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return (None, None, None)

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Bifurcation {junction_id!r} must define exactly three angles.")

    parsed: list[float | None] = []
    for angle in value:
        if angle is None:
            parsed.append(None)
            continue

        text = str(angle).strip().lower()
        if text in {"none", "null", ""}:
            parsed.append(None)
            continue

        parsed.append(float(angle))

    return parsed[0], parsed[1], parsed[2]
