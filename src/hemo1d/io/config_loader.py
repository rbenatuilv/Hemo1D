from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    path = Path(path)
    with path.open("r") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top level in {path}.")

    return data
