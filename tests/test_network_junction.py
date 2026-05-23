import math

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointData, EndpointSide, StateAtPoint
from hemo1d.boundary.junction import (
    JunctionData,
    JunctionEndpointData,
    JunctionResidual,
    JunctionSolver,
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


def make_rest_three_vessel_junction_data() -> JunctionData:
    physics = make_physics()

    inlet = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.RIGHT,
        name="inlet",
    )
    outlet1 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.LEFT,
        name="outlet1",
    )
    outlet2 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.LEFT,
        name="outlet2",
    )

    return JunctionData(endpoints=(inlet, outlet1, outlet2))


def make_rest_two_vessel_junction_data() -> JunctionData:
    physics = make_physics()

    upstream = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.RIGHT,
        name="upstream",
    )
    downstream = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(physics.params.area0, 0.0),
        side=EndpointSide.LEFT,
        name="downstream",
    )

    return JunctionData(endpoints=(upstream, downstream))


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


def test_junction_residual_rejects_non_positive_dt():
    data = make_rest_three_vessel_junction_data()

    with pytest.raises(ValueError):
        JunctionResidual(data=data, dt=0.0)


def test_three_vessel_junction_initial_guess_uses_previous_endpoint_values():
    data = make_rest_three_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    x0 = residual.initial_guess()

    expected = np.array(
        [
            data.endpoints[0].endpoint_data.state.area,
            data.endpoints[0].endpoint_data.state.flow_rate,
            data.endpoints[1].endpoint_data.state.area,
            data.endpoints[1].endpoint_data.state.flow_rate,
            data.endpoints[2].endpoint_data.state.area,
            data.endpoints[2].endpoint_data.state.flow_rate,
        ]
    )

    assert x0.shape == (6,)
    assert np.allclose(x0, expected)


def test_two_vessel_junction_initial_guess_uses_previous_endpoint_values():
    data = make_rest_two_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    x0 = residual.initial_guess()

    expected = np.array(
        [
            data.endpoints[0].endpoint_data.state.area,
            data.endpoints[0].endpoint_data.state.flow_rate,
            data.endpoints[1].endpoint_data.state.area,
            data.endpoints[1].endpoint_data.state.flow_rate,
        ]
    )

    assert x0.shape == (4,)
    assert np.allclose(x0, expected)


def test_three_vessel_junction_residual_is_zero_at_rest_and_has_6x6_jacobian():
    data = make_rest_three_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    x = residual.initial_guess()
    r = residual(x)
    J = residual.jacobian(x)

    assert np.allclose(r, 0.0, rtol=1.0e-12, atol=1.0e-12)
    assert J.shape == (6, 6)


def test_two_vessel_junction_residual_is_zero_at_rest_and_has_4x4_jacobian():
    data = make_rest_two_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    x = residual.initial_guess()
    r = residual(x)
    J = residual.jacobian(x)

    assert np.allclose(r, 0.0, rtol=1.0e-12, atol=1.0e-12)
    assert J.shape == (4, 4)


def test_three_vessel_junction_solver_returns_rest_state_at_rest():
    data = make_rest_three_vessel_junction_data()

    solver = JunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-5)

    A0 = data.endpoints[0].physics.params.area0

    assert solution.newton_result.converged
    assert len(solution.endpoint_states) == 3
    assert math.isclose(solution.endpoint_states[0].area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[1].area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[2].area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)

    assert math.isclose(solution.endpoint_states[0].flow_rate, 0.0, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[1].flow_rate, 0.0, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[2].flow_rate, 0.0, abs_tol=1.0e-12)


def test_two_vessel_junction_solver_returns_rest_state_at_rest():
    data = make_rest_two_vessel_junction_data()

    solver = JunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-5)

    A0 = data.endpoints[0].physics.params.area0

    assert solution.newton_result.converged
    assert len(solution.endpoint_states) == 2
    assert math.isclose(solution.endpoint_states[0].area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[1].area, A0, rel_tol=1.0e-12, abs_tol=1.0e-12)

    assert math.isclose(solution.endpoint_states[0].flow_rate, 0.0, abs_tol=1.0e-12)
    assert math.isclose(solution.endpoint_states[1].flow_rate, 0.0, abs_tol=1.0e-12)


def test_three_vessel_junction_solver_conserves_mass_for_small_flow():
    """
    Construct previous-time endpoint states with a small inlet flow and split
    outlet flows. The solved junction should satisfy mass conservation.
    """
    physics = make_physics()
    A0 = physics.params.area0

    inlet = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 2.0e-4),
        side=EndpointSide.RIGHT,
        name="inlet",
    )
    outlet1 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 1.0e-4),
        side=EndpointSide.LEFT,
        name="outlet1",
    )
    outlet2 = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 1.0e-4),
        side=EndpointSide.LEFT,
        name="outlet2",
    )

    data = JunctionData(endpoints=(inlet, outlet1, outlet2))

    solver = JunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-6)

    assert solution.newton_result.converged

    mass_residual = (
        solution.endpoint_states[0].flow_rate
        - solution.endpoint_states[1].flow_rate
        - solution.endpoint_states[2].flow_rate
    )

    assert math.isclose(mass_residual, 0.0, rel_tol=1.0e-10, abs_tol=1.0e-12)

    assert solution.endpoint_states[0].area > 0.0
    assert solution.endpoint_states[1].area > 0.0
    assert solution.endpoint_states[2].area > 0.0


def test_two_vessel_junction_solver_conserves_mass_for_small_flow():
    physics = make_physics()
    A0 = physics.params.area0

    upstream = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 2.0e-4),
        side=EndpointSide.RIGHT,
        name="upstream",
    )
    downstream = JunctionEndpointData(
        physics=physics,
        endpoint_data=make_endpoint_data(A0, 2.0e-4),
        side=EndpointSide.LEFT,
        name="downstream",
    )

    data = JunctionData(endpoints=(upstream, downstream))

    solver = JunctionSolver()
    solution = solver.solve(data=data, dt=1.0e-6)

    assert solution.newton_result.converged

    mass_residual = solution.endpoint_states[0].flow_rate - solution.endpoint_states[1].flow_rate

    assert math.isclose(mass_residual, 0.0, rel_tol=1.0e-10, abs_tol=1.0e-12)
    assert solution.endpoint_states[0].area > 0.0
    assert solution.endpoint_states[1].area > 0.0


def test_three_vessel_junction_residual_rejects_wrong_shape():
    data = make_rest_three_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    with pytest.raises(ValueError):
        residual(np.zeros(4))


def test_two_vessel_junction_residual_rejects_wrong_shape():
    data = make_rest_two_vessel_junction_data()
    residual = JunctionResidual(data=data, dt=1.0e-5)

    with pytest.raises(ValueError):
        residual(np.zeros(5))
