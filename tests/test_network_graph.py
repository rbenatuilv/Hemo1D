import numpy as np
import pytest

from hemo1d.boundary import CopyBoundaryCondition, NonReflectingBoundary
from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig
from hemo1d.solvers.cg.factory import create_cg_vessel
from hemo1d.solvers.vessel import Vessel
from hemo1d.topology import (
    Junction,
    NetworkEndpoint,
    VascularNetwork,
)


def make_physics(length: float = 2.0, area0: float = 0.126) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=length, area0=area0, beta=0.060606e7),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_cg_vessel(vessel_id: str) -> Vessel:
    physics = make_physics()
    discretization = CGFEMDiscretization(
        CGMeshConfig(
            length=physics.params.length,
            num_cells=8,
            degree=1,
        )
    )

    vessel = create_cg_vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=discretization,
    )
    vessel.interpolate_rest_state()

    return vessel


def test_network_endpoint_outward_flow_signs():
    left = NetworkEndpoint("v1", EndpointSide.LEFT)
    right = NetworkEndpoint("v1", EndpointSide.RIGHT)

    assert left.outward_normal_sign == -1
    assert right.outward_normal_sign == 1

    assert left.outward_flow(2.0) == -2.0
    assert right.outward_flow(2.0) == 2.0

    assert left.label() == "v1.left"
    assert right.label() == "v1.right"


def test_create_cg_vessel_rest_state():
    vessel = make_cg_vessel("v1")

    assert vessel.vessel_id == "v1"
    assert np.allclose(vessel.state_n.A.x.array, vessel.physics.params.area0)
    assert np.allclose(vessel.state_n.Q.x.array, 0.0)
    assert vessel.compute_stable_dt(cfl=0.25) > 0.0


def test_single_vessel_network_is_complete():
    vessel = make_cg_vessel("vessel")

    left = NetworkEndpoint("vessel", EndpointSide.LEFT)
    right = NetworkEndpoint("vessel", EndpointSide.RIGHT)

    network = VascularNetwork(
        vessels={"vessel": vessel},
        external_boundaries={
            left: CopyBoundaryCondition(),
            right: NonReflectingBoundary(),
        },
    )

    assert network.is_complete()
    assert network.unassigned_endpoints() == set()
    assert network.vessel_ids() == ["vessel"]


def test_three_vessel_network_is_complete():
    parent = make_cg_vessel("parent")
    daughter1 = make_cg_vessel("daughter1")
    daughter2 = make_cg_vessel("daughter2")

    junction = Junction(
        endpoints=(
            NetworkEndpoint("parent", EndpointSide.RIGHT),
            NetworkEndpoint("daughter1", EndpointSide.LEFT),
            NetworkEndpoint("daughter2", EndpointSide.LEFT),
        )
    )

    network = VascularNetwork(
        vessels={
            "parent": parent,
            "daughter1": daughter1,
            "daughter2": daughter2,
        },
        junctions=[junction],
        external_boundaries={
            NetworkEndpoint("parent", EndpointSide.LEFT): CopyBoundaryCondition(),
            NetworkEndpoint("daughter1", EndpointSide.RIGHT): NonReflectingBoundary(),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    assert network.is_complete()
    assert network.unassigned_endpoints() == set()


def test_two_vessel_junction_network_is_complete():
    upstream = make_cg_vessel("upstream")
    downstream = make_cg_vessel("downstream")

    junction = Junction(
        endpoints=(
            NetworkEndpoint("upstream", EndpointSide.RIGHT),
            NetworkEndpoint("downstream", EndpointSide.LEFT),
        )
    )

    network = VascularNetwork(
        vessels={
            "upstream": upstream,
            "downstream": downstream,
        },
        junctions=[junction],
        external_boundaries={
            NetworkEndpoint("upstream", EndpointSide.LEFT): CopyBoundaryCondition(),
            NetworkEndpoint("downstream", EndpointSide.RIGHT): NonReflectingBoundary(),
        },
    )

    assert network.is_complete()
    assert network.unassigned_endpoints() == set()


def test_network_rejects_vessel_key_mismatch():
    vessel = make_cg_vessel("actual_id")

    with pytest.raises(ValueError):
        VascularNetwork(
            vessels={"wrong_id": vessel},
        )


def test_network_rejects_unknown_junction_vessel():
    parent = make_cg_vessel("parent")

    junction = Junction(
        endpoints=(
            NetworkEndpoint("parent", EndpointSide.RIGHT),
            NetworkEndpoint("missing1", EndpointSide.LEFT),
            NetworkEndpoint("missing2", EndpointSide.LEFT),
        )
    )

    with pytest.raises(ValueError):
        VascularNetwork(
            vessels={"parent": parent},
            junctions=[junction],
        )


def test_network_rejects_unknown_boundary_vessel():
    vessel = make_cg_vessel("vessel")

    with pytest.raises(ValueError):
        VascularNetwork(
            vessels={"vessel": vessel},
            external_boundaries={
                NetworkEndpoint("unknown", EndpointSide.LEFT): CopyBoundaryCondition(),
            },
        )


def test_network_rejects_duplicate_junction_endpoint():
    parent = make_cg_vessel("parent")
    daughter1 = make_cg_vessel("daughter1")
    daughter2 = make_cg_vessel("daughter2")

    repeated_endpoint = NetworkEndpoint("parent", EndpointSide.RIGHT)

    junction_1 = Junction(
        endpoints=(
            repeated_endpoint,
            NetworkEndpoint("daughter1", EndpointSide.LEFT),
            NetworkEndpoint("daughter2", EndpointSide.LEFT),
        )
    )

    junction_2 = Junction(
        endpoints=(
            repeated_endpoint,
            NetworkEndpoint("daughter1", EndpointSide.RIGHT),
            NetworkEndpoint("daughter2", EndpointSide.RIGHT),
        )
    )

    with pytest.raises(ValueError):
        VascularNetwork(
            vessels={
                "parent": parent,
                "daughter1": daughter1,
                "daughter2": daughter2,
            },
            junctions=[junction_1, junction_2],
        )


def test_network_rejects_endpoint_as_boundary_and_junction():
    parent = make_cg_vessel("parent")
    daughter1 = make_cg_vessel("daughter1")
    daughter2 = make_cg_vessel("daughter2")

    endpoint = NetworkEndpoint("parent", EndpointSide.RIGHT)

    junction = Junction(
        endpoints=(
            endpoint,
            NetworkEndpoint("daughter1", EndpointSide.LEFT),
            NetworkEndpoint("daughter2", EndpointSide.LEFT),
        )
    )

    with pytest.raises(ValueError):
        VascularNetwork(
            vessels={
                "parent": parent,
                "daughter1": daughter1,
                "daughter2": daughter2,
            },
            junctions=[junction],
            external_boundaries={
                endpoint: CopyBoundaryCondition(),
            },
        )
