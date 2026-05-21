# tests/test_physics.py

import math

import numpy as np
import pytest

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics


@pytest.fixture
def params() -> ModelParameters:
    return ModelParameters(
        blood=BloodParameters(
            rho=1.06,
            mu=0.035,
        ),
        vessel=VesselParameters(
            length=10.0,
            area0=0.126,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=85.0,
        p_ext=5.0,
    )


@pytest.fixture
def physics(params: ModelParameters) -> Hemo1DPhysics:
    return Hemo1DPhysics(params, NP_BACKEND)


def test_model_parameters_shortcuts(params: ModelParameters):
    assert params.rho == params.blood.rho
    assert params.mu == params.blood.mu
    assert params.length == params.vessel.length
    assert params.area0 == params.vessel.area0
    assert params.beta == params.vessel.beta
    assert params.gamma == params.gamma_profile


def test_alpha_from_gamma_profile(params: ModelParameters):
    gamma = params.gamma_profile
    expected_alpha = (gamma + 2.0) / (gamma + 1.0)

    assert math.isclose(params.alpha, expected_alpha)
    assert math.isclose(params.alpha, 4.0 / 3.0)


def test_psi_matches_tube_law_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    A0 = params.area0
    beta = params.beta

    expected = beta * (math.sqrt(A) - math.sqrt(A0)) / A0

    assert math.isclose(physics.psi(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_psi_is_zero_at_reference_area(physics: Hemo1DPhysics, params: ModelParameters):
    assert math.isclose(physics.psi(params.area0), 0.0, abs_tol=1e-14)


def test_pressure_adds_reference_and_external_pressure(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.10 * params.area0

    expected = params.p_ext + params.p0 + physics.psi(A)

    assert math.isclose(physics.pressure(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_pressure_at_reference_area_is_p0_plus_pext(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    expected = params.p0 + params.p_ext

    assert math.isclose(physics.pressure(params.area0), expected, rel_tol=1e-14)


def test_dpsi_dA_matches_analytic_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    A0 = params.area0
    beta = params.beta

    expected = beta / (2.0 * A0 * math.sqrt(A))

    assert math.isclose(physics.dpsi_dA(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_dpsi_dA_matches_finite_difference(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    eps = 1.0e-7 * params.area0

    finite_difference = (physics.psi(A + eps) - physics.psi(A - eps)) / (2.0 * eps)

    assert math.isclose(physics.dpsi_dA(A), finite_difference, rel_tol=1e-7, abs_tol=1e-7)


def test_velocity_is_Q_over_A(physics: Hemo1DPhysics):
    A = 2.0
    Q = 6.0

    assert physics.velocity(A, Q) == 3.0


def test_wave_speed_matches_definition(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0

    expected = math.sqrt(A * physics.dpsi_dA(A) / params.rho)

    assert math.isclose(physics.wave_speed(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_wave_speed_matches_simplified_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    A0 = params.area0
    beta = params.beta
    rho = params.rho

    expected = math.sqrt(beta * math.sqrt(A) / (2.0 * rho * A0))

    assert math.isclose(physics.wave_speed(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_wave_speed_is_positive_for_positive_area(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0

    assert physics.wave_speed(A) > 0.0


def test_c_alpha_matches_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    Q = 0.10

    alpha = params.alpha
    u = Q / A
    c = physics.wave_speed(A)

    expected = math.sqrt(c * c + alpha * (alpha - 1.0) * u * u)

    assert math.isclose(physics.c_alpha(A, Q), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_eigenvalues_match_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    Q = 0.10

    alpha = params.alpha
    u = Q / A
    ca = physics.c_alpha(A, Q)

    expected_plus = alpha * u + ca
    expected_minus = alpha * u - ca

    lam_plus, lam_minus = physics.eigenvalues(A, Q)

    assert math.isclose(lam_plus, expected_plus, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(lam_minus, expected_minus, rel_tol=1e-14, abs_tol=1e-14)


def test_eigenvalues_are_opposite_sign_at_rest(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = params.area0
    Q = 0.0

    lam_plus, lam_minus = physics.eigenvalues(A, Q)

    assert lam_plus > 0.0
    assert lam_minus < 0.0
    assert math.isclose(lam_plus, -lam_minus, rel_tol=1e-14, abs_tol=1e-14)


def test_left_eigenvectors_match_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    Q = 0.10

    alpha = params.alpha
    u = Q / A
    ca = physics.c_alpha(A, Q)

    l_plus, l_minus = physics.left_eigenvectors(A, Q)

    assert math.isclose(l_plus[0], ca - alpha * u, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(l_plus[1], 1.0, rel_tol=1e-14, abs_tol=1e-14)

    assert math.isclose(l_minus[0], -ca - alpha * u, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(l_minus[1], 1.0, rel_tol=1e-14, abs_tol=1e-14)


def test_friction_coefficient_matches_formula(physics: Hemo1DPhysics, params: ModelParameters):
    gamma = params.gamma
    mu = params.mu
    rho = params.rho

    expected = 2.0 * (gamma + 2.0) * math.pi * mu / rho

    assert math.isclose(
        physics.friction_coefficient(),
        expected,
        rel_tol=1e-14,
        abs_tol=1e-14,
    )


def test_friction_coefficient_for_gamma_2_is_8_pi_mu_over_rho(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    expected = 8.0 * math.pi * params.mu / params.rho

    assert math.isclose(
        physics.friction_coefficient(),
        expected,
        rel_tol=1e-14,
        abs_tol=1e-14,
    )


def test_C1_matches_integrated_wave_speed_formula(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0
    A0 = params.area0
    beta = params.beta
    rho = params.rho

    expected = beta * (A * math.sqrt(A) - A0 * math.sqrt(A0)) / (3.0 * rho * A0)

    assert math.isclose(physics.C1(A), expected, rel_tol=1e-14, abs_tol=1e-14)


def test_C1_is_zero_at_reference_area(physics: Hemo1DPhysics, params: ModelParameters):
    assert math.isclose(physics.C1(params.area0), 0.0, abs_tol=1e-14)


def test_C1_derivative_matches_wave_speed_squared(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0
    eps = 1.0e-7 * params.area0

    finite_difference = (physics.C1(A + eps) - physics.C1(A - eps)) / (2.0 * eps)
    c_squared = physics.wave_speed(A) ** 2

    assert math.isclose(finite_difference, c_squared, rel_tol=1e-7, abs_tol=1e-7)


def test_flux_matches_conservative_formula(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    Q = 0.10

    F = physics.flux(A, Q)

    expected_F1 = Q
    expected_F2 = params.alpha * Q * Q / A + physics.C1(A)

    assert np.shape(F) == (2,)
    assert math.isclose(F[0], expected_F1, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(F[1], expected_F2, rel_tol=1e-14, abs_tol=1e-14)


def test_flux_at_rest_has_zero_mass_flux_and_zero_C1(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = params.area0
    Q = 0.0

    F = physics.flux(A, Q)

    assert math.isclose(F[0], 0.0, abs_tol=1e-14)
    assert math.isclose(F[1], 0.0, abs_tol=1e-14)


def test_flux_accepts_numpy_arrays(physics: Hemo1DPhysics, params: ModelParameters):
    A = np.array([params.area0, 1.10 * params.area0, 1.25 * params.area0])
    Q = np.array([0.0, 0.05, 0.10])

    F = physics.flux(A, Q)

    assert F.shape == (2, 3)
    assert np.all(np.isfinite(F))


def test_source_matches_constant_vessel_formula(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0
    Q = 0.10

    S = physics.source(A, Q)

    expected_S1 = 0.0
    expected_S2 = physics.friction_coefficient() * Q / A

    assert np.shape(S) == (2,)
    assert math.isclose(S[0], expected_S1, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(S[1], expected_S2, rel_tol=1e-14, abs_tol=1e-14)


def test_source_is_zero_at_rest(physics: Hemo1DPhysics, params: ModelParameters):
    A = params.area0
    Q = 0.0

    S = physics.source(A, Q)

    assert math.isclose(S[0], 0.0, abs_tol=1e-14)
    assert math.isclose(S[1], 0.0, abs_tol=1e-14)


def test_H_matrix_matches_quasilinear_matrix(physics: Hemo1DPhysics, params: ModelParameters):
    A = 1.25 * params.area0
    Q = 0.10

    H = physics.H_matrix(A, Q)

    alpha = params.alpha
    u = Q / A
    c = physics.wave_speed(A)

    expected = np.array(
        [
            [0.0, 1.0],
            [c * c - alpha * u * u, 2.0 * alpha * u],
        ]
    )

    assert H.shape == (2, 2)
    assert np.allclose(H, expected, rtol=1e-14, atol=1e-14)


def test_H_matrix_eigenvalues_match_physics_eigenvalues(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0
    Q = 0.10

    H = physics.H_matrix(A, Q)

    eigvals = np.linalg.eigvals(H)
    lam_plus, lam_minus = physics.eigenvalues(A, Q)

    expected = np.sort(np.array([lam_minus, lam_plus]))
    actual = np.sort(eigvals)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_total_pressure_matches_dimensional_formula(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = 1.25 * params.area0
    Q = 0.10

    u = Q / A

    expected = physics.pressure(A) + 0.5 * params.rho * u * u

    assert math.isclose(
        physics.total_pressure(A, Q, include_density=True),
        expected,
        rel_tol=1e-14,
        abs_tol=1e-14,
    )


def test_total_pressure_without_density_option(
    physics: Hemo1DPhysics,
):
    A = 0.2
    Q = 0.1

    u = Q / A

    expected = physics.pressure(A) + 0.5 * u * u

    assert math.isclose(
        physics.total_pressure(A, Q, include_density=False),
        expected,
        rel_tol=1e-14,
        abs_tol=1e-14,
    )


def test_total_pressure_increases_with_flow_rate(
    physics: Hemo1DPhysics,
    params: ModelParameters,
):
    A = params.area0

    ptot_0 = physics.total_pressure(A, 0.0)
    ptot_1 = physics.total_pressure(A, 0.10)

    assert ptot_1 > ptot_0