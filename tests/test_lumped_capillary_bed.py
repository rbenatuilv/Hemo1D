import math

import numpy as np
import pytest

from hemo1d.boundary import NonReflectingBoundary, PrescribedFlowBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.lumped import CapillaryBedEndpoint, LumpedCapillaryBed
from hemo1d.solvers.dg import DGFEMDiscretization, DGMeshConfig
from hemo1d.solvers.dg.factory import create_dg_vessel
from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.time import TimeConfig
from hemo1d.topology import Junction, NetworkEndpoint, VascularNetwork


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_vessel(vessel_id: str):
    physics = make_physics()
    disc = DGFEMDiscretization(
        DGMeshConfig(
            length=physics.params.length,
            num_cells=16,
            degree=1,
        )
    )
    vessel = create_dg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=disc,
        time_scheme="euler",
    )
    vessel.interpolate_rest_state()
    return vessel


def make_bed(endpoint: NetworkEndpoint, bed_id: str = "bed") -> LumpedCapillaryBed:
    return LumpedCapillaryBed(
        bed_id=bed_id,
        endpoints=[
            CapillaryBedEndpoint(
                endpoint=endpoint,
                resistance=1.0e6,
            )
        ],
        compliance=1.0e-7,
        venous_resistance=1.0e6,
        venous_pressure=0.0,
        pressure=0.0,
    )


def assert_finite_endpoint_states(states):
    for state in states.values():
        assert math.isfinite(state.area)
        assert state.area > 0.0
        assert math.isfinite(state.flow_rate)


def assert_bed_balances(bed: LumpedCapillaryBed, states, vessels, pressure_old, dt):
    endpoint_resistances = {
        bed_endpoint.endpoint: bed_endpoint.resistance
        for bed_endpoint in bed.endpoints
    }

    for endpoint, state in states.items():
        vessel = vessels[endpoint.vessel_id]
        inflow = endpoint.outward_flow(state.flow_rate)
        pressure_1d = vessel.physics.pressure(state.area)
        resistance = endpoint_resistances[endpoint]

        assert inflow == pytest.approx(
            (pressure_1d - bed.pressure) / resistance,
            abs=1.0e-8,
        )

    assert bed.compliance * (bed.pressure - pressure_old) / dt == pytest.approx(
        bed.last_total_inflow - bed.last_venous_outflow,
        abs=1.0e-8,
    )


def test_single_endpoint_bed_solves_with_zero_pressure_scale():
    vessel = make_vessel("vessel")
    endpoint = NetworkEndpoint("vessel", EndpointSide.RIGHT)
    bed = make_bed(endpoint)

    states = bed.solve(vessels={"vessel": vessel}, dt=1.0e-5)

    assert set(states) == {endpoint}
    assert math.isfinite(bed.pressure)
    assert_finite_endpoint_states(states)


def test_single_endpoint_bed_solves_with_nonzero_pressure_scale():
    vessel = make_vessel("vessel")
    endpoint = NetworkEndpoint("vessel", EndpointSide.RIGHT)
    bed = LumpedCapillaryBed(
        bed_id="nonzero_pressure",
        endpoints=[
            CapillaryBedEndpoint(
                endpoint=endpoint,
                resistance=1.0e6,
            )
        ],
        compliance=1.0e-7,
        venous_resistance=1.0e6,
        venous_pressure=500.0,
        pressure=800.0,
    )

    states = bed.solve(vessels={"vessel": vessel}, dt=1.0e-5)

    assert set(states) == {endpoint}
    assert math.isfinite(bed.pressure)
    assert_finite_endpoint_states(states)


def test_single_endpoint_bed_runs_for_a_few_steps():
    vessel = make_vessel("vessel")
    outlet = NetworkEndpoint("vessel", EndpointSide.RIGHT)
    bed = make_bed(outlet)

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            NetworkEndpoint("vessel", EndpointSide.LEFT): PrescribedFlowBoundary(
                lambda t: 1.0e-4
            ),
        },
        lumped_beds=[bed],
    )
    solver = NetworkSolver(network)

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=3.0e-5,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
        show_progress=False,
    )

    final_vessel = result.network.vessels["vessel"]
    _, area, flow_rate = final_vessel.state_arrays()

    assert np.all(np.isfinite(area))
    assert np.all(area > 0.0)
    assert np.all(np.isfinite(flow_rate))
    assert math.isfinite(bed.pressure)


def test_shared_bed_solve_returns_all_endpoint_states_and_diagnostics():
    vessel_1 = make_vessel("v1")
    vessel_2 = make_vessel("v2")
    vessels = {"v1": vessel_1, "v2": vessel_2}
    endpoint_1 = NetworkEndpoint("v1", EndpointSide.RIGHT)
    endpoint_2 = NetworkEndpoint("v2", EndpointSide.RIGHT)
    bed = LumpedCapillaryBed(
        bed_id="shared",
        endpoints=[
            CapillaryBedEndpoint(endpoint=endpoint_1, resistance=1.0e6),
            CapillaryBedEndpoint(endpoint=endpoint_2, resistance=2.0e6),
        ],
        compliance=1.0e-7,
        venous_resistance=1.0e6,
        venous_pressure=100.0,
        pressure=400.0,
    )
    pressure_old = bed.pressure
    dt = 1.0e-5

    states = bed.solve(
        vessels=vessels,
        dt=dt,
    )

    assert set(states) == {endpoint_1, endpoint_2}
    assert math.isfinite(bed.pressure)
    assert_finite_endpoint_states(states)
    assert bed.last_total_inflow == pytest.approx(
        sum(bed.last_endpoint_inflows.values())
    )
    assert bed.last_venous_outflow == pytest.approx(
        (bed.pressure - bed.venous_pressure) / bed.venous_resistance
    )
    assert_bed_balances(
        bed=bed,
        states=states,
        vessels=vessels,
        pressure_old=pressure_old,
        dt=dt,
    )


def test_network_rejects_endpoint_as_boundary_and_lumped_bed():
    vessel = make_vessel("vessel")
    endpoint = NetworkEndpoint("vessel", EndpointSide.RIGHT)

    with pytest.raises(ValueError, match="external boundaries and lumped bed"):
        VascularNetwork(
            vessels={"vessel": vessel},
            external_boundaries={endpoint: NonReflectingBoundary()},
            lumped_beds=[make_bed(endpoint)],
        )


def test_network_rejects_endpoint_in_two_lumped_beds():
    vessel = make_vessel("vessel")
    endpoint = NetworkEndpoint("vessel", EndpointSide.RIGHT)

    with pytest.raises(ValueError, match="more than one lumped bed"):
        VascularNetwork(
            vessels={"vessel": vessel},
            lumped_beds=[
                make_bed(endpoint, bed_id="bed_1"),
                make_bed(endpoint, bed_id="bed_2"),
            ],
        )


def test_network_rejects_endpoint_as_junction_and_lumped_bed():
    upstream = make_vessel("upstream")
    downstream = make_vessel("downstream")
    endpoint = NetworkEndpoint("upstream", EndpointSide.RIGHT)
    junction = Junction(
        endpoints=(
            endpoint,
            NetworkEndpoint("downstream", EndpointSide.LEFT),
        )
    )

    with pytest.raises(ValueError, match="junction and lumped bed"):
        VascularNetwork(
            vessels={
                "upstream": upstream,
                "downstream": downstream,
            },
            junctions=[junction],
            lumped_beds=[make_bed(endpoint)],
        )
