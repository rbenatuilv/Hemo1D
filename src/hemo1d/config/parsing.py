from __future__ import annotations

from typing import Any

from hemo1d.config.models import BloodConfig, JunctionConfig, VesselConfig
from hemo1d.config.sides import parse_endpoint_side
from hemo1d.topology.endpoint import NetworkEndpoint


def parse_vessels(data: Any, *, defaults: dict[str, Any]) -> dict[str, VesselConfig]:
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((required_string(item, "id"), item) for item in data)
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
            length=float(pick(raw, "length")),
            area0=float(pick(raw, "area0", "initial_area", "A0")),
            beta=float(pick(raw, "beta", "beta_coeff")),
            left_bound=optional_label(raw.get("left_bound", raw.get("left_boundary"))),
            right_bound=optional_label(raw.get("right_bound", raw.get("right_boundary"))),
            blood=blood,
            gamma_profile=float(raw.get("gamma_profile", defaults.get("gamma_profile", 2.0))),
            p0=float(raw.get("p0", defaults.get("p0", 0.0))),
            p_ext=float(raw.get("p_ext", defaults.get("p_ext", 0.0))),
            gamma_pressure_loss=float(
                raw.get(
                    "gamma_pressure_loss",
                    raw.get(
                        "gamma_pressure",
                        defaults.get("gamma_pressure_loss", 0.0),
                    ),
                )
            ),
        )

    return vessels


def parse_junctions(data: Any) -> list[JunctionConfig]:
    if data in ({}, [], None):
        return []

    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((required_string(item, "id"), item) for item in data)
    else:
        raise ValueError("'junctions' must be a JSON object or list.")

    junctions: list[JunctionConfig] = []
    for junction_id, raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"Junction {junction_id!r} must be a JSON object.")

        if "branches" in raw:
            branches = raw["branches"]
            positions = raw["positions"]
            if len(branches) not in (2, 3) or len(positions) != len(branches):
                raise ValueError(f"Junction {junction_id!r} must have 2 or 3 branches/positions.")
            endpoints = tuple(
                NetworkEndpoint(str(branch), parse_endpoint_side(position))
                for branch, position in zip(branches, positions, strict=True)
            )
            angles = parse_angles(
                raw.get("angles"),
                junction_id=junction_id,
                count=len(endpoints),
            )
        elif "endpoints" in raw:
            if not isinstance(raw["endpoints"], (list, tuple)):
                raise ValueError(f"Junction {junction_id!r} endpoints must be a list.")
            endpoints = tuple(parse_endpoint_ref(endpoint) for endpoint in raw["endpoints"])
            angles = parse_angles(
                raw.get("angles"),
                junction_id=junction_id,
                count=len(endpoints),
            )
        else:
            raise ValueError(
                f"Junction {junction_id!r} must define branches/positions or endpoints."
            )

        junctions.append(
            JunctionConfig(
                junction_id=str(junction_id),
                endpoints=endpoints,
                angles=angles,
            )
        )

    return junctions


def parse_endpoint_ref(data: Any) -> NetworkEndpoint:
    if not isinstance(data, dict):
        raise ValueError("Endpoint reference must be an object with vessel_id and side.")

    vessel_id = data.get("vessel_id", data.get("vessel", data.get("id")))
    if vessel_id is None:
        raise ValueError("Endpoint reference is missing vessel_id.")

    return NetworkEndpoint(str(vessel_id), parse_endpoint_side(data.get("side")))


def pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise ValueError(f"Missing required field; expected one of {keys}.")


def required_string(data: dict[str, Any], key: str) -> str:
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"List entries must contain {key!r}.")
    return str(data[key])


def optional_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    return label or None


def parse_angles(
    value: Any,
    *,
    junction_id: str,
    count: int = 3,
) -> tuple[float | None, ...]:
    if value is None:
        return (None,) * count

    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"Junction {junction_id!r} must define exactly {count} angles.")

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

    return tuple(parsed)


__all__ = [
    "optional_label",
    "parse_angles",
    "parse_endpoint_ref",
    "parse_junctions",
    "parse_vessels",
    "pick",
    "required_string",
]
