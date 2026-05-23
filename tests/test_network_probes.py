import math

import pytest
from mpi4py import MPI

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.vessel import Vessel
from hemo1d.topology import (
    Junction,
    NetworkEndpoint,
    VascularNetwork,
)
from hemo1d.observe import NetworkProbeRecorder, ProbePoint
from hemo1d.solvers.time import TimeConfig


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=0.060606e7),
        gamma_profile=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_vessel(vessel_id: str, physics: Hemo1DPhysics, num_cells: int = 8) -> Vessel:
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=num_cells, degree=1)
    )
    vessel = create_cg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=disc,
    )
    vessel.interpolate_rest_state()
    return vessel


def make_network() -> VascularNetwork:
    parent = make_vessel("parent", make_physics())
    d1 = make_vessel("daughter1", make_physics())
    d2 = make_vessel("daughter2", make_physics())

    junction = Junction(
        endpoints=(
            NetworkEndpoint("parent", EndpointSide.RIGHT),
            NetworkEndpoint("daughter1", EndpointSide.LEFT),
            NetworkEndpoint("daughter2", EndpointSide.LEFT),
        )
    )

    return VascularNetwork(
        vessels={
            "parent": parent,
            "daughter1": d1,
            "daughter2": d2,
        },
        junctions=[junction],
        external_boundaries={
            NetworkEndpoint("parent", EndpointSide.LEFT): PrescribedFlowBoundary(lambda t: 0.0),
            NetworkEndpoint("daughter1", EndpointSide.RIGHT): NonReflectingBoundary(),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )


def test_network_probe_recorder_samples_rest_state():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network probe test is serial-only for now.")

    network = make_network()

    probes = [
        ProbePoint(vessel_id="parent", name="inlet", coordinate=0.0),
        ProbePoint(vessel_id="parent", name="junction", coordinate=2.0),
        ProbePoint(vessel_id="daughter1", name="junction", coordinate=0.0),
        ProbePoint(vessel_id="daughter2", name="junction", coordinate=0.0),
    ]

    recorder = NetworkProbeRecorder(
        vessels=network.vessels,
        probes=probes,
    )

    samples = recorder.sample(time=0.0)

    assert len(samples) == len(probes)

    for sample in samples:
        assert math.isclose(sample.area, 0.126)
        assert math.isclose(sample.flow_rate, 0.0)


def test_network_probe_recorder_rejects_unknown_vessel():
    network = make_network()

    with pytest.raises(ValueError):
        NetworkProbeRecorder(
            vessels=network.vessels,
            probes=[ProbePoint(vessel_id="unknown", name="bad", coordinate=0.0)],
        )


def test_general_network_solver_records_probe_history():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Network probe test is serial-only for now.")

    network = make_network()
    solver = NetworkSolver(network)

    probes = [
        ProbePoint(vessel_id="parent", name="inlet", coordinate=0.0),
        ProbePoint(vessel_id="parent", name="junction", coordinate=2.0),
        ProbePoint(vessel_id="daughter1", name="junction", coordinate=0.0),
        ProbePoint(vessel_id="daughter2", name="junction", coordinate=0.0),
    ]

    result = solver.run(
        config=TimeConfig(t0=0.0, t_end=2.0e-5, fixed_dt=1.0e-5),
        record_every=1,
        probes=probes,
    )

    assert len(result.history.probes.samples) == 3 * 4
    assert ("parent", "inlet") in result.history.probes.keys()
    assert ("daughter1", "junction") in result.history.probes.keys()
