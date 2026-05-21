import math

import numpy as np
import pytest
from mpi4py import MPI

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointSide, StateAtPoint
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig, CGState


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(
            length=2.0,
            area0=0.126,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_cg_mesh_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        CGMeshConfig(length=0.0, num_cells=8)

    with pytest.raises(ValueError):
        CGMeshConfig(length=1.0, num_cells=0)

    with pytest.raises(ValueError):
        CGMeshConfig(length=1.0, num_cells=8, degree=0)


def test_create_cg_discretization(physics: Hemo1DPhysics):
    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    assert disc.domain.topology.dim == 1
    assert disc.domain.geometry.dim == 1
    assert disc.length == physics.params.length
    assert disc.num_cells == 8
    assert disc.degree == 1


def test_create_state_returns_cg_state(physics: Hemo1DPhysics):
    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    state = disc.create_state(name="test")

    assert isinstance(state, CGState)
    assert state.A.name == "test_A"
    assert state.Q.name == "test_Q"


def test_h_min_for_uniform_mesh(physics: Hemo1DPhysics):
    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    assert math.isclose(disc.h_min(), physics.params.length / 8.0)


def test_num_dofs_for_degree_1_serial(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Serial-only dof count test.")

    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    # Linear CG on an interval with N cells has N+1 scalar dofs in serial.
    assert disc.num_dofs() == 9


def test_endpoint_dofs_serial(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Serial-only endpoint dof test.")

    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    left = disc.endpoint_dofs(EndpointSide.LEFT)
    right = disc.endpoint_dofs(EndpointSide.RIGHT)

    assert len(left) == 1
    assert len(right) == 1
    assert left[0] != right[0]


def test_interpolate_rest_state_serial(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Serial-only endpoint value test.")

    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    left_state = disc.endpoint_state(state, EndpointSide.LEFT)
    right_state = disc.endpoint_state(state, EndpointSide.RIGHT)

    assert isinstance(left_state, StateAtPoint)
    assert isinstance(right_state, StateAtPoint)

    assert math.isclose(left_state.area, physics.params.area0)
    assert math.isclose(right_state.area, physics.params.area0)
    assert math.isclose(left_state.flow_rate, 0.0)
    assert math.isclose(right_state.flow_rate, 0.0)

    assert np.allclose(state.A.x.array, physics.params.area0)
    assert np.allclose(state.Q.x.array, 0.0)


def test_set_endpoint_state_serial(physics: Hemo1DPhysics):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Serial-only endpoint value test.")

    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    left_boundary = BoundaryState(area=0.2, flow_rate=1.5)
    right_boundary = BoundaryState(area=0.3, flow_rate=-0.7)

    disc.set_endpoint_state(state, EndpointSide.LEFT, left_boundary)
    disc.set_endpoint_state(state, EndpointSide.RIGHT, right_boundary)

    left_state = disc.endpoint_state(state, EndpointSide.LEFT)
    right_state = disc.endpoint_state(state, EndpointSide.RIGHT)

    assert math.isclose(left_state.area, left_boundary.area)
    assert math.isclose(left_state.flow_rate, left_boundary.flow_rate)

    assert math.isclose(right_state.area, right_boundary.area)
    assert math.isclose(right_state.flow_rate, right_boundary.flow_rate)


def test_state_copy_from(physics: Hemo1DPhysics):
    config = CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    disc = CGFEMDiscretization(config)

    source = disc.create_state(name="source")
    target = disc.create_state(name="target")

    source.A.x.array[:] = 2.0
    source.Q.x.array[:] = -1.0
    source.scatter_forward()

    target.copy_from(source)

    assert np.allclose(target.A.x.array, source.A.x.array)
    assert np.allclose(target.Q.x.array, source.Q.x.array)