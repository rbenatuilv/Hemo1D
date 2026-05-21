from hemo1d.core.state import BoundaryState, EndpointSide, StateAtPoint, VesselEndpoint


def test_state_at_point_stores_area_and_flow_rate():
    state = StateAtPoint(area=0.126, flow_rate=1.5)

    assert state.area == 0.126
    assert state.flow_rate == 1.5


def test_boundary_state_stores_area_and_flow_rate():
    state = BoundaryState(area=0.2, flow_rate=-0.1)

    assert state.area == 0.2
    assert state.flow_rate == -0.1


def test_left_endpoint_has_negative_outward_sign():
    endpoint = VesselEndpoint(vessel_id="v1", side=EndpointSide.LEFT)

    assert endpoint.outward_normal_sign == -1


def test_right_endpoint_has_positive_outward_sign():
    endpoint = VesselEndpoint(vessel_id="v1", side=EndpointSide.RIGHT)

    assert endpoint.outward_normal_sign == 1


def test_left_endpoint_outward_flow_is_minus_Q():
    endpoint = VesselEndpoint(vessel_id="v1", side=EndpointSide.LEFT)

    assert endpoint.outward_flow(2.0) == -2.0


def test_right_endpoint_outward_flow_is_plus_Q():
    endpoint = VesselEndpoint(vessel_id="v1", side=EndpointSide.RIGHT)

    assert endpoint.outward_flow(2.0) == 2.0