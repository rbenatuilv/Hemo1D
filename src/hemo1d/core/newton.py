from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


ResidualFunction = Callable[[np.ndarray], np.ndarray]
JacobianFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NewtonConfig:
    """
    Configuration for Newton's method.

    residual_tol:
        Absolute tolerance for ||R(x)||_2.

    increment_tol:
        Relative tolerance for ||dx||_2 / (||x||_2 + increment_scale).

    increment_scale:
        Small positive number used to avoid division by zero in the relative
        increment criterion.

    max_iterations:
        Maximum number of Newton iterations.

    finite_difference_eps:
        Base perturbation size for finite-difference Jacobian columns.

    damping:
        Multiplicative damping factor for the Newton update.
        For now use 1.0. Later we can add line search.
    """

    residual_tol: float = 1.0e-10
    increment_tol: float = 1.0e-10
    increment_scale: float = 1.0
    max_iterations: int = 100
    finite_difference_eps: float = 1.0e-7
    damping: float = 1.0

    def __post_init__(self) -> None:
        if self.residual_tol <= 0.0:
            raise ValueError("residual_tol must be positive.")
        if self.increment_tol <= 0.0:
            raise ValueError("increment_tol must be positive.")
        if self.increment_scale <= 0.0:
            raise ValueError("increment_scale must be positive.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.finite_difference_eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive.")
        if self.damping <= 0.0 or self.damping > 1.0:
            raise ValueError("damping must be in (0, 1].")


@dataclass(frozen=True)
class NewtonResult:
    """
    Result of a Newton solve.
    """

    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    increment_norm: float
    message: str


def finite_difference_jacobian(
    residual: ResidualFunction,
    x: np.ndarray,
    eps: float = 1.0e-7,
) -> np.ndarray:
    """
    Approximate the Jacobian of residual(x) using centered finite differences.

    Column j is:

        (R(x + h e_j) - R(x - h e_j)) / (2h)

    with h scaled by the magnitude of x_j.
    """
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(residual(x), dtype=float)

    if r0.ndim != 1:
        raise ValueError("residual(x) must return a one-dimensional array.")

    n_unknowns = x.size
    n_equations = r0.size

    jacobian = np.zeros((n_equations, n_unknowns), dtype=float)

    for j in range(n_unknowns):
        h = eps * max(1.0, abs(x[j]))

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += h
        x_minus[j] -= h

        r_plus = np.asarray(residual(x_plus), dtype=float)
        r_minus = np.asarray(residual(x_minus), dtype=float)

        jacobian[:, j] = (r_plus - r_minus) / (2.0 * h)

    return jacobian


class NewtonSolver:
    """
    Generic Newton solver for small nonlinear systems.

    The first use case is the 1D hemodynamics junction system, which has only
    six unknowns for a bifurcation. Therefore a dense NumPy linear solve is
    appropriate.
    """

    def __init__(self, config: NewtonConfig | None = None) -> None:
        self.config = config if config is not None else NewtonConfig()

    def solve(
        self,
        residual: ResidualFunction,
        x0: np.ndarray,
        jacobian: JacobianFunction | None = None,
        raise_on_failure: bool = False,
    ) -> NewtonResult:
        """
        Solve R(x) = 0.

        If jacobian is None, a centered finite-difference Jacobian is used.
        """
        x = np.asarray(x0, dtype=float).copy()

        if x.ndim != 1:
            raise ValueError("x0 must be a one-dimensional array.")

        cfg = self.config

        residual_norm = float("inf")
        increment_norm = float("inf")

        for iteration in range(cfg.max_iterations + 1):
            r = np.asarray(residual(x), dtype=float)

            if r.ndim != 1:
                raise ValueError("residual(x) must return a one-dimensional array.")

            residual_norm = float(np.linalg.norm(r, ord=2))

            if residual_norm <= cfg.residual_tol:
                return NewtonResult(
                    x=x,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    increment_norm=increment_norm,
                    message="Converged by residual norm.",
                )

            if iteration == cfg.max_iterations:
                break

            if jacobian is None:
                J = finite_difference_jacobian(
                    residual=residual,
                    x=x,
                    eps=cfg.finite_difference_eps,
                )
            else:
                J = np.asarray(jacobian(x), dtype=float)

            if J.ndim != 2:
                raise ValueError("Jacobian must be a two-dimensional array.")

            if J.shape[0] != r.size or J.shape[1] != x.size:
                raise ValueError(
                    f"Jacobian shape {J.shape} incompatible with "
                    f"residual size {r.size} and unknown size {x.size}."
                )

            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError as exc:
                result = NewtonResult(
                    x=x,
                    converged=False,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    increment_norm=increment_norm,
                    message=f"Linear solve failed: {exc}",
                )
                if raise_on_failure:
                    raise RuntimeError(result.message) from exc
                return result

            dx = cfg.damping * dx
            x = x + dx

            increment_norm = float(
                np.linalg.norm(dx, ord=2)
                / (np.linalg.norm(x, ord=2) + cfg.increment_scale)
            )

            if increment_norm <= cfg.increment_tol:
                r_new = np.asarray(residual(x), dtype=float)
                residual_norm = float(np.linalg.norm(r_new, ord=2))

                return NewtonResult(
                    x=x,
                    converged=residual_norm <= 10.0 * cfg.residual_tol,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    increment_norm=increment_norm,
                    message="Converged by relative increment."
                    if residual_norm <= 10.0 * cfg.residual_tol
                    else "Increment small but residual still above tolerance.",
                )

        result = NewtonResult(
            x=x,
            converged=False,
            iterations=cfg.max_iterations,
            residual_norm=residual_norm,
            increment_norm=increment_norm,
            message="Maximum Newton iterations reached.",
        )

        if raise_on_failure:
            raise RuntimeError(result.message)

        return result