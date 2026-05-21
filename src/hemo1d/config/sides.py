from __future__ import annotations

from hemo1d.core.state import EndpointSide


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


__all__ = ["parse_endpoint_side"]
