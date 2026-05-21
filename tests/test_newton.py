import math

import numpy as np
import pytest

from hemo1d.core.newton import NewtonConfig, NewtonSolver, finite_difference_jacobian


def test_newton_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        NewtonConfig(residual_tol=0.0)

    with pytest.raises(ValueError):
        NewtonConfig(increment_tol=0.0)

    with pytest.raises(ValueError):
        NewtonConfig(increment_scale=0.0)

    with pytest.raises(ValueError):
        NewtonConfig(max_iterations=0)

    with pytest.raises(ValueError):
        NewtonConfig(finite_difference_eps=0.0)

    with pytest.raises(ValueError):
        NewtonConfig(damping=0.0)

    with pytest.raises(ValueError):
        NewtonConfig(damping=1.5)


def test_finite_difference_jacobian_scalar_equation():
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 2.0])

    x = np.array([2.0])

    J = finite_difference_jacobian(residual, x)

    assert J.shape == (1, 1)
    assert math.isclose(J[0, 0], 4.0, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_newton_solves_scalar_square_root_problem():
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 2.0])

    solver = NewtonSolver(
        NewtonConfig(
            residual_tol=1.0e-12,
            increment_tol=1.0e-12,
            max_iterations=20,
        )
    )

    result = solver.solve(residual=residual, x0=np.array([1.0]))

    assert result.converged
    assert math.isclose(result.x[0], math.sqrt(2.0), rel_tol=1.0e-12, abs_tol=1.0e-12)
    assert result.residual_norm < 1.0e-12


def test_newton_solves_2x2_nonlinear_system_with_fd_jacobian():
    """
    Solve:
        x^2 + y^2 = 5
        x - y = 1

    Solution near the initial guess is:
        x = 2, y = 1
    """

    def residual(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array(
            [
                x * x + y * y - 5.0,
                x - y - 1.0,
            ]
        )

    solver = NewtonSolver(
        NewtonConfig(
            residual_tol=1.0e-12,
            increment_tol=1.0e-12,
            max_iterations=20,
        )
    )

    result = solver.solve(residual=residual, x0=np.array([1.5, 0.5]))

    assert result.converged
    assert np.allclose(result.x, np.array([2.0, 1.0]), rtol=1.0e-12, atol=1.0e-12)


def test_newton_solves_with_analytic_jacobian():
    def residual(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array(
            [
                x * x + y * y - 5.0,
                x - y - 1.0,
            ]
        )

    def jacobian(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array(
            [
                [2.0 * x, 2.0 * y],
                [1.0, -1.0],
            ]
        )

    solver = NewtonSolver(
        NewtonConfig(
            residual_tol=1.0e-12,
            increment_tol=1.0e-12,
            max_iterations=20,
        )
    )

    result = solver.solve(
        residual=residual,
        jacobian=jacobian,
        x0=np.array([1.5, 0.5]),
    )

    assert result.converged
    assert np.allclose(result.x, np.array([2.0, 1.0]), rtol=1.0e-12, atol=1.0e-12)


def test_newton_reports_failure_when_max_iterations_too_small():
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 2.0])

    solver = NewtonSolver(
        NewtonConfig(
            residual_tol=1.0e-14,
            increment_tol=1.0e-14,
            max_iterations=1,
        )
    )

    result = solver.solve(residual=residual, x0=np.array([10.0]))

    assert not result.converged
    assert result.message == "Maximum Newton iterations reached."


def test_newton_raise_on_failure():
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 - 2.0])

    solver = NewtonSolver(
        NewtonConfig(
            residual_tol=1.0e-14,
            increment_tol=1.0e-14,
            max_iterations=1,
        )
    )

    with pytest.raises(RuntimeError):
        solver.solve(
            residual=residual,
            x0=np.array([10.0]),
            raise_on_failure=True,
        )