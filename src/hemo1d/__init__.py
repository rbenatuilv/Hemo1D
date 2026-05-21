"""
Hemo1D public API.

Use the package through the high-level model facade:

    import hemo1d as hd

    model = hd.load_from_config("data/network.json")
    model.set_inlet(vessel_id="BAS", kind="velocity", function=v_in)
    model.set_solver(method="DG", h=0.1, dt=1e-4, poly_order=1)
    results = model.solve(t_end=1.0)
"""

from __future__ import annotations

from hemo1d.api import HemodynamicModel, NetworkModel, load_from_config
from hemo1d.boundary import (
    NonReflectingBoundary,
    PrescribedAreaBoundary,
    PrescribedFlowBoundary,
    PrescribedPressureBoundary,
    create_positive_sine_inflow,
    create_pulsatile_inflow,
    create_sinusoidal_inflow,
)
from hemo1d.config import (
    BloodConfig,
    JunctionConfig,
    NetworkConfig,
    VesselConfig,
    load_network_config,
)
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide, StateAtPoint
from hemo1d.io import (
    CSVScalarFunction,
    VelocityInflowSeries,
    load_json,
    make_flow_rate_from_velocity_csv,
    read_area_csv,
    read_flow_rate_csv,
    read_velocity_csv,
    write_diagnostics_csv,
    write_probe_history_csv,
    write_vessel_final_state_csv,
)
from hemo1d.results import Results

__version__ = "0.1.0"

__all__ = [
    "BloodConfig",
    "BloodParameters",
    "CSVScalarFunction",
    "EndpointSide",
    "HemodynamicModel",
    "Hemo1DPhysics",
    "JunctionConfig",
    "ModelParameters",
    "NetworkConfig",
    "NetworkModel",
    "NonReflectingBoundary",
    "PrescribedAreaBoundary",
    "PrescribedFlowBoundary",
    "PrescribedPressureBoundary",
    "Results",
    "StateAtPoint",
    "VelocityInflowSeries",
    "VesselConfig",
    "VesselParameters",
    "create_positive_sine_inflow",
    "create_pulsatile_inflow",
    "create_sinusoidal_inflow",
    "load_from_config",
    "load_json",
    "load_network_config",
    "make_flow_rate_from_velocity_csv",
    "read_area_csv",
    "read_flow_rate_csv",
    "read_velocity_csv",
    "write_diagnostics_csv",
    "write_probe_history_csv",
    "write_vessel_final_state_csv",
]
