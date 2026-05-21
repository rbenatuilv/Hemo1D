from __future__ import annotations

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.dg import DGFEMDiscretization, DGMeshConfig


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(
            length=1.0,
            area0=0.126,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_dg_degree_one_mesh_data_shapes() -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=2.0, num_cells=4, degree=1)
    )

    assert discretization.num_cells == 4
    assert discretization.degree == 1
    assert discretization.num_local_dofs == 2
    assert discretization.coordinates().shape == (4, 2)
    assert discretization.basis_at_quad.shape == (3, 2)
    assert discretization.basis_derivative_at_quad.shape == (3, 2)
    assert discretization.mass_matrix.shape == (2, 2)
    assert discretization.inverse_mass_matrix.shape == (2, 2)


def test_dg_degree_zero_mesh_data_shapes() -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=2.0, num_cells=4, degree=0)
    )

    assert discretization.num_cells == 4
    assert discretization.degree == 0
    assert discretization.num_local_dofs == 1
    assert discretization.coordinates().shape == (4, 1)
    assert discretization.basis_at_quad.shape == (1, 1)
    assert discretization.basis_derivative_at_quad.shape == (1, 1)
    assert discretization.mass_matrix.shape == (1, 1)


def test_rest_state_interpolation_degree_one(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=8, degree=1)
    )
    state = discretization.create_state(name="n")

    discretization.interpolate_rest_state(state, physics)

    np.testing.assert_allclose(state.A, physics.params.area0)
    np.testing.assert_allclose(state.Q, 0.0)


def test_endpoint_state_degree_one(
    physics: Hemo1DPhysics,
) -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")
    discretization.interpolate_rest_state(state, physics)

    state.A[0, 0] = 1.1
    state.Q[0, 0] = 2.1
    state.A[-1, 1] = 1.2
    state.Q[-1, 1] = 2.2

    left = discretization.endpoint_state(state, EndpointSide.LEFT)
    right = discretization.endpoint_state(state, EndpointSide.RIGHT)

    assert left.area == pytest.approx(1.1)
    assert left.flow_rate == pytest.approx(2.1)
    assert right.area == pytest.approx(1.2)
    assert right.flow_rate == pytest.approx(2.2)


def test_endpoint_derivative_degree_one_linear_function() -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=2.0, num_cells=2, degree=1)
    )
    state = discretization.create_state(name="n")

    # A(x) = 3 + 2x
    # Q(x) = -1 + 5x
    x = discretization.coordinates()
    state.A[:, :] = 3.0 + 2.0 * x
    state.Q[:, :] = -1.0 + 5.0 * x

    left_dA, left_dQ = discretization.endpoint_derivatives(state, EndpointSide.LEFT)
    right_dA, right_dQ = discretization.endpoint_derivatives(state, EndpointSide.RIGHT)

    assert left_dA == pytest.approx(2.0)
    assert left_dQ == pytest.approx(5.0)
    assert right_dA == pytest.approx(2.0)
    assert right_dQ == pytest.approx(5.0)


def test_interpolate_state_degree_one() -> None:
    discretization = DGFEMDiscretization(
        DGMeshConfig(length=1.0, num_cells=4, degree=1)
    )
    state = discretization.create_state(name="n")

    discretization.interpolate_state(
        state,
        area_fn=lambda x: 1.0 + x,
        flow_rate_fn=lambda x: 2.0 - x,
    )

    x = discretization.coordinates()

    np.testing.assert_allclose(state.A, 1.0 + x)
    np.testing.assert_allclose(state.Q, 2.0 - x)


def test_invalid_degree_raises() -> None:
    with pytest.raises(NotImplementedError):
        DGMeshConfig(length=1.0, num_cells=4, degree=2)