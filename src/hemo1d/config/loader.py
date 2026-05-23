from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hemo1d.config.models import NetworkConfig
from hemo1d.config.parsing import parse_junctions, parse_vessels
from hemo1d.config.validation import validate_network_config


def load_network_config(path: str | Path) -> NetworkConfig:
    """
    Load a network configuration from JSON.

    Supported forms:
        - a combined file with top-level ``vessels`` and optional ``junctions`` key;
        - the existing project ``vessels.json`` shape, optionally paired with a
          sibling ``junctions.json`` file.
    """

    config_path = Path(path)
    data = load_json_object(config_path)

    if "vessels" in data:
        if "bifurcations" in data:
            raise ValueError("Use 'junctions' instead of obsolete 'bifurcations'.")
        vessel_data = data["vessels"]
        junction_data = data.get("junctions", {})
        defaults = data.get("defaults", {})
    else:
        vessel_data = data
        junction_data = load_sibling_junctions(config_path)
        defaults = {}

    vessels = parse_vessels(vessel_data, defaults=defaults)
    junctions = parse_junctions(junction_data)

    config = NetworkConfig(
        vessels=vessels,
        junctions=junctions,
        source_path=config_path,
    )
    validate_network_config(config)
    return config


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top level in {path}.")

    return data


def load_sibling_junctions(path: Path) -> Any:
    sibling = path.with_name("junctions.json")
    if sibling.exists():
        return load_json_object(sibling)
    return {}


__all__ = [
    "load_json_object",
    "load_network_config",
    "load_sibling_junctions",
]
