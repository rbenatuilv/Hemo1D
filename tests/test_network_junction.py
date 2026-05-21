import math

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint
from hemo1d.boundary.junction import (
    BifurcationJunctionData,
    BifurcationJunctionResidual,
    BifurcationJunctionSolver,
    JunctionEndpointData,
    compatibility_target,
    outgoing_left_eigenvector,
)


def make_physics(area0: float = 0.126, beta: float = 0.060606e7) -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(length=2.0, area0=area0, beta=beta),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def make_endpoint_data(area: float, flow_rate: float) -> EndpointData:
    return EndpointData(
        state=StateAtPoint(area=area, flow_rate=flow_rate),
        d_area_dz=0.0,
        d_flow_rate_dz=0.0,
    )


def make_rest_junction_data() -> BifurcationJunctionData:
    physics = make_physics()

    parent = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.RIGHT,
        name="parent",
    )
    daughter1 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.LEFT,
        name="daughter1",
    )
    daughter2 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.LEFT,
        name="daughter2",
    )

    return BifurcationJunctionData(
        parent=parent,
        daughter1=daughter1,
        daughter2=daughter2,
    )


def test_compatibility_target_at_rest_is_rest_state():
    physics = make_physics()
    data = make_endpoint_data(physics.params.area0, 0.0)

    cc = compatibility_target(
        physics=physics,
        endpoint_data=data,
        dt=1.0e-5,
    )

    assert np.allclose(cc, np.array([physics.params.area0, 0.0]))


def test_outgoing_eigenvector_right_is_l_plus():
    physics = make_physics()
    data = make_endpoint_data(physics.params.area0, 0.0)

    l_plus, _ = physics.left_eigenvectors(physics.params.area0, 0.0)

    l_out = outgoing_left_eigenvector(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.RIGHT,
    )

    assert np.allclose(l_out, np.array(l_plus, dtype=float))


def test_outgoing_eigenvector_left_is_l_minus():
    physics = make_physics()
    data = make_endpoint_data(physics.params.area0, 0.0)

    _, l_minus = physics.left_eigenvectors(physics.params.area0, 0.0)

    l_out = outgoing_left_eigenvector(
        physics=physics,
        endpoint_data=data,
        side=EndpointSide.LEFT,
    )

    assert np.allclose(l_out, np.array(l_minus, dtype=float))


def test_bifurcation_residual_rejects_non_positive_dt():
    data = make_rest_junction_data()

    with pytest.raises(ValueError):
        BifurcationJunctionResidual(data=data, dt=0.0)


def test_bifurcation_initial_guess_uses_previous_endpoint_values():
    data = make_rest_junction_data()
    residual = BifurcationJunctionResidual(data=data, dt=1.0e-5)

    x0 = residual.initial_guess()

    expected = np.array(
        [
            data.parent.endpoint_data.state.area,
            data.parent.endpoint_data.state.flow_rate,
            data.daughter1.endpoint_data.state.area,
            data.daughter1.endpoint_data.state.flow_rate,
            data.daughter2.endpoint_data.state.area,
            data.daughter2.endpoint_data.state.flow_rate,
        ]
    )

    assert np.allclose(x0, expected)


def test_bifurcation_residual_is_zero_at_rest():
    data = make_rest_junction_data()
    residual = BifurcationJunctionResidual(data=data, dt=1.0e-5)

    x = residual.initial_guess()
    r = residual(x)

    assert np.allclose(r, 0.0, rtol=1.0e-12, atol=1.0e-12)


def test_bifurcation_solver_returns_rest_state_at_rest():
    data = make_rest_junction_data()

    solver = BifurcationJunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-5)

    A0 = data.parent.physics.params.area0

    assert solution.newton_result.converged
    assert math.isclose(solution.parent.area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(solution.daughter1.area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(solution.daughter2.area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)

    assert math.isclose(solution.parent.flow_rate, 0.0, abs_tol=1.0e-12)
    assert math.isclose(solution.daughter1.flow_rate, 0.0, abs_tol=1.0e-12)
    assert math.isclose(solution.daughter2.flow_rate, 0.0, abs_tol=1.0e-12)


def test_bifurcation_solver_conserves_mass_for_small_parent_flow():
    """
    Construct previous-time endpoint states with a small parent flow and split
    daughter flows. The solved junction should satisfy mass conservation.
    """
    physics = make_physics()
    A0 = physics.params.area0

    parent = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 2.0e-4),
        side=EndpointSide.RIGHT,
        name="parent",
    )
    daughter1 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 1.0e-4),
        side=EndpointSide.LEFT,
        name="daughter1",
    )
    daughter2 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 1.0e-4),
        side=EndpointSide.LEFT,
        name="daughter2",
    )

    data = BifurcationJunctionData(
        parent=parent,
        daughter1=daughter1,
        daughter2=daughter2,
    )

    solver = BifurcationJunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-6)

    assert solution.newton_result.converged

    mass_residual = (
        solution.parent.flow_rate
        - solution.daughter1.flow_rate
        - solution.daughter2.flow_rate
    )

    assert math.isclose(mass_residual, 0.0, rel_tol=1.0e-10, abs_tol=1.0e-12)

    assert solution.parent.area > 0.0
    assert solution.daughter1.area > 0.0
    assert solution.daughter2.area > 0.0


def test_bifurcation_residual_rejects_wrong_shape():
    data = make_rest_junction_data()
    residual = BifurcationJunctionResidual(data=data, dt=1.0e-5)

    with pytest.raises(ValueError):
        residual(np.zeros(5))
