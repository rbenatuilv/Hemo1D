import csv

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.io import (
    write_diagnostics_csv,
    write_probe_history_csv,
    write_vessel_final_state_csv,
)
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.topology import NetworkEndpoint, VascularNetwork
from hemo1d.observe import ProbePoint
from hemo1d.solvers.time import TimeConfig


def make_small_result():
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
    )
    physics = Hemo1DPhysics(params, NP_BACKEND)

    disc = CGFEMDiscretization(
        CGMeshConfig(length=params.length, num_cells=8, degree=1)
    )

    vessel = create_cg_vessel(
        vessel_id="vessel",
        physics=physics,
        discretization=disc,
    )
    vessel.interpolate_rest_state()

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(
                lambda t: 0.0
            ),
            NetworkEndpoint("vessel", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(t0=0.0, t_end=2.0e-5, fixed_dt=1.0e-5),
        probes=[
            ProbePoint(vessel_id="vessel", name="left", coordinate=0.0),
            ProbePoint(vessel_id="vessel", name="right", coordinate=params.length),
        ],
    )

    final_vessel = result.network.vessels["vessel"]

    return final_vessel, result


def test_write_diagnostics_csv(tmp_path):
    _, result = make_small_result()

    path = tmp_path / "diagnostics.csv"
    write_diagnostics_csv(result.history, path)

    assert path.exists()

    with path.open() as file:
        rows = list(csv.DictReader(file))

    # Network diagnostics: one row per time per vessel.
    assert len(rows) == len(result.history.diagnostics)
    assert "time" in rows[0]
    assert "vessel_id" in rows[0]
    assert "min_area" in rows[0]
    assert "max_pressure" in rows[0]


def test_write_probe_history_csv(tmp_path):
    _, result = make_small_result()

    path = tmp_path / "probes.csv"
    write_probe_history_csv(result.history, path)

    assert path.exists()

    with path.open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == len(result.history.probes.samples)
    assert "time" in rows[0]
    assert "vessel_id" in rows[0]
    assert "name" in rows[0]
    assert "area" in rows[0]
    assert "flow_rate" in rows[0]


def test_write_vessel_final_state_csv(tmp_path):
    vessel, _ = make_small_result()

    path = tmp_path / "final_state.csv"
    write_vessel_final_state_csv(vessel, path)

    assert path.exists()

    with path.open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == vessel.num_dofs
    assert "z" in rows[0]
    assert "area" in rows[0]
    assert "flow_rate" in rows[0]
