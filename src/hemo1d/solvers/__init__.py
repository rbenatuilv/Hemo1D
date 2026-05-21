from __future__ import annotations

from importlib import import_module
from typing import Any

from hemo1d.solvers.base import (
    BaseSolver,
    VesselDiscretization,
    VesselStateArrayExtractor,
    VesselStateSampler,
    VesselStepper,
)
from hemo1d.solvers.time import TimeConfig

_LAZY_EXPORTS = {
    "NetworkSolver": ("hemo1d.solvers.model_solver", "NetworkSolver"),
    "NetworkSolverResult": ("hemo1d.solvers.model_solver", "NetworkSolverResult"),
    "Vessel": ("hemo1d.solvers.vessel", "Vessel"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'hemo1d.solvers' has no attribute {name!r}")


__all__ = [
    "BaseSolver",
    "NetworkSolver",
    "NetworkSolverResult",
    "TimeConfig",
    "Vessel",
    "VesselDiscretization",
    "VesselStateArrayExtractor",
    "VesselStateSampler",
    "VesselStepper",
]
