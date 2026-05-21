from __future__ import annotations

from importlib import import_module
from typing import Any

from hemo1d.observe.diagnostics import (
    StateDiagnostics,
    compute_cg_state_diagnostics,
    compute_state_diagnostics_from_arrays,
    compute_vessel_diagnostics,
)
from hemo1d.observe.history import ProbeHistory, ProbePoint, ProbeSample
from hemo1d.observe.probes import NetworkProbeRecorder

_LAZY_EXPORTS = {
    "CGProbeRecorder": ("hemo1d.observe.cg", "CGProbeRecorder"),
    "evaluate_cg_scalar_nearest_dof": (
        "hemo1d.observe.cg",
        "evaluate_cg_scalar_nearest_dof",
    ),
    "sample_cg_state_at_probe": ("hemo1d.observe.cg", "sample_cg_state_at_probe"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'hemo1d.observe' has no attribute {name!r}")


__all__ = [
    "CGProbeRecorder",
    "NetworkProbeRecorder",
    "ProbeHistory",
    "ProbePoint",
    "ProbeSample",
    "StateDiagnostics",
    "compute_cg_state_diagnostics",
    "compute_state_diagnostics_from_arrays",
    "compute_vessel_diagnostics",
    "evaluate_cg_scalar_nearest_dof",
    "sample_cg_state_at_probe",
]
