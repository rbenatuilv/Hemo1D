import math

import pytest
from mpi4py import MPI

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.observe import CGProbeRecorder, ProbePoint


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=0.126, beta=0.060606e7),
        gamma_profile=2.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_probe_rejects_point_outside_domain(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )

    with pytest.raises(ValueError):
        CGProbeRecorder(
            vessel_id="vessel",
            discretization=disc,
            physics=physics,
            probes=[ProbePoint(vessel_id="vessel", name="bad", coordinate=3.0)],
        )


def test_probe_rejects_wrong_vessel_id(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )

    with pytest.raises(ValueError):
        CGProbeRecorder(
            vessel_id="vessel",
            discretization=disc,
            physics=physics,
            probes=[ProbePoint(vessel_id="other", name="left", coordinate=0.0)],
        )


def test_probe_samples_rest_state(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Probe recorder is serial-only for now.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    recorder = CGProbeRecorder(
        vessel_id="vessel",
        discretization=disc,
        physics=physics,
        probes=[
            ProbePoint(vessel_id="vessel", name="left", coordinate=0.0),
            ProbePoint(vessel_id="vessel", name="mid", coordinate=physics.params.length / 2.0),
            ProbePoint(vessel_id="vessel", name="right", coordinate=physics.params.length),
        ],
    )

    samples = recorder.sample(state, time=0.5)

    assert len(samples) == 3

    for sample in samples:
        assert sample.vessel_id == "vessel"
        assert math.isclose(sample.time, 0.5)
        assert math.isclose(sample.area, physics.params.area0)
        assert math.isclose(sample.flow_rate, 0.0)
        assert math.isclose(sample.pressure, physics.pressure(physics.params.area0))