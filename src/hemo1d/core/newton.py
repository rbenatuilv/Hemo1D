from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


ResidualFunction = Callable[[np.ndarray], np.ndarray]
JacobianFunction = Callable[[np.ndarray], np.ndarray]
ResidualNormFunction = Callable[[np.ndarray], float]
CandidateValidator = Callable[[np.ndarray], bool]


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

    line_search:
        If True, backtrack the Newton update until the residual decreases.

    min_step_factor:
        Smallest line-search multiplier to try before declaring failure.

    relaxed_residual_tol:
        Optional tolerance for accepting a stalled but already near-converged
        iterate when line search cannot find a strict residual decrease.

    use_increment_criterion:
        If True, allow convergence by the relative increment criterion.
    """

    residual_tol: float = 1.0e-10
    increment_tol: float = 1.0e-10
    increment_scale: float = 1.0
    max_iterations: int = 100
    finite_difference_eps: float = 1.0e-7
    damping: float = 1.0
    line_search: bool = False
    min_step_factor: float = 1.0e-8
    relaxed_residual_tol: float | None = None
    use_increment_criterion: bool = True

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
        if self.min_step_factor <= 0.0 or self.min_step_factor > 1.0:
            raise ValueError("min_step_factor must be in (0, 1].")
        if self.relaxed_residual_tol is not None and self.relaxed_residual_tol <= 0.0:
            raise ValueError("relaxed_residual_tol must be positive when provided.")


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
    *,
    variable_scales: np.ndarray | None = None,
    method: str = "centered",
    r0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Approximate the Jacobian of residual(x) using finite differences.

    By default, column j is centered:

        (R(x + h e_j) - R(x - h e_j)) / (2h)

    In forward mode, column j is:

        (R(x + h e_j) - R(x)) / h

    with h scaled by the magnitude of x_j and optional variable scales.
    """
    x = np.asarray(x, dtype=float)
    if method not in {"centered", "forward"}:
        raise ValueError("method must be 'centered' or 'forward'.")

    if r0 is None:
        r0 = np.asarray(residual(x), dtype=float)
    else:
        r0 = np.asarray(r0, dtype=float)

    if r0.ndim != 1:
        raise ValueError("residual(x) must return a one-dimensional array.")

    if variable_scales is not None:
        variable_scales = np.asarray(variable_scales, dtype=float)
        if variable_scales.shape != x.shape:
            raise ValueError("variable_scales must have the same shape as x.")

    n_unknowns = x.size
    n_equations = r0.size

    jacobian = np.zeros((n_equations, n_unknowns), dtype=float)

    for j in range(n_unknowns):
        variable_scale = max(1.0, abs(float(x[j])))
        if variable_scales is not None:
            variable_scale = max(variable_scale, abs(float(variable_scales[j])))

        h = eps * variable_scale

        x_plus = x.copy()
        x_plus[j] += h

        r_plus = np.asarray(residual(x_plus), dtype=float)

        if method == "forward":
            jacobian[:, j] = (r_plus - r0) / h
        else:
            x_minus = x.copy()
            x_minus[j] -= h

            r_minus = np.asarray(residual(x_minus), dtype=float)
            jacobian[:, j] = (r_plus - r_minus) / (2.0 * h)

    return jacobian


class NewtonSolver:
    """
    Generic Newton solver for small nonlinear systems.

    The first use case is the 1D hemodynamics junction system, which has only
    a handful of unknowns. Therefore a dense NumPy linear solve is appropriate.
    """

    def __init__(self, config: NewtonConfig | None = None) -> None:
        self.config = config if config is not None else NewtonConfig()

    def solve(
        self,
        residual: ResidualFunction,
        x0: np.ndarray,
        jacobian: JacobianFunction | None = None,
        raise_on_failure: bool = False,
        residual_norm: ResidualNormFunction | None = None,
        is_valid_candidate: CandidateValidator | None = None,
    ) -> NewtonResult:
        """
        Solve R(x) = 0.

        If jacobian is None, a centered finite-difference Jacobian is used.
        """
        x = np.asarray(x0, dtype=float).copy()

        if x.ndim != 1:
            raise ValueError("x0 must be a one-dimensional array.")

        cfg = self.config
        norm = residual_norm if residual_norm is not None else _euclidean_norm

        residual_norm_value = float("inf")
        increment_norm = float("inf")

        for iteration in range(cfg.max_iterations + 1):
            r = np.asarray(residual(x), dtype=float)

            if r.ndim != 1:
                raise ValueError("residual(x) must return a one-dimensional array.")

            residual_norm_value = float(norm(r))

            if residual_norm_value <= cfg.residual_tol:
                return NewtonResult(
                    x=x,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm_value,
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
                    r0=r,
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
                    residual_norm=residual_norm_value,
                    increment_norm=increment_norm,
                    message=f"Linear solve failed: {exc}",
                )
                if raise_on_failure:
                    raise RuntimeError(result.message) from exc
                return result

            dx = cfg.damping * dx
            step_result = self._take_step(
                residual=residual,
                x=x,
                dx=dx,
                current_residual_norm=residual_norm_value,
                norm=norm,
                is_valid_candidate=is_valid_candidate,
                iteration=iteration,
            )
            if not step_result.converged:
                result = NewtonResult(
                    x=x,
                    converged=False,
                    iterations=iteration,
                    residual_norm=residual_norm_value,
                    increment_norm=increment_norm,
                    message=step_result.message,
                )
                if raise_on_failure:
                    raise RuntimeError(result.message)
                return result

            if step_result.accepted_solution:
                return NewtonResult(
                    x=step_result.x,
                    converged=True,
                    iterations=iteration,
                    residual_norm=residual_norm_value,
                    increment_norm=0.0,
                    message=step_result.message,
                )

            x = step_result.x
            dx_used = step_result.increment

            increment_norm = float(
                np.linalg.norm(dx_used, ord=2)
                / (np.linalg.norm(x, ord=2) + cfg.increment_scale)
            )

            if cfg.use_increment_criterion and increment_norm <= cfg.increment_tol:
                r_new = np.asarray(residual(x), dtype=float)
                residual_norm_value = float(norm(r_new))

                return NewtonResult(
                    x=x,
                    converged=residual_norm_value <= 10.0 * cfg.residual_tol,
                    iterations=iteration + 1,
                    residual_norm=residual_norm_value,
                    increment_norm=increment_norm,
                    message="Converged by relative increment."
                    if residual_norm_value <= 10.0 * cfg.residual_tol
                    else "Increment small but residual still above tolerance.",
                )

        result = NewtonResult(
            x=x,
            converged=False,
            iterations=cfg.max_iterations,
            residual_norm=residual_norm_value,
            increment_norm=increment_norm,
            message="Maximum Newton iterations reached.",
        )

        if raise_on_failure:
            raise RuntimeError(result.message)

        return result

    def _take_step(
        self,
        *,
        residual: ResidualFunction,
        x: np.ndarray,
        dx: np.ndarray,
        current_residual_norm: float,
        norm: ResidualNormFunction,
        is_valid_candidate: CandidateValidator | None,
        iteration: int,
    ) -> _StepResult:
        cfg = self.config

        if not cfg.line_search:
            candidate = x + dx
            if is_valid_candidate is not None and not is_valid_candidate(candidate):
                return _StepResult(
                    x=x,
                    increment=np.zeros_like(dx),
                    converged=False,
                    message="Newton candidate failed validity check.",
                )
            return _StepResult(
                x=candidate,
                increment=dx,
                converged=True,
                message="Step accepted.",
            )

        alpha = 1.0
        while alpha >= cfg.min_step_factor:
            candidate_dx = alpha * dx
            candidate = x + candidate_dx
            if is_valid_candidate is not None and not is_valid_candidate(candidate):
                alpha *= 0.5
                continue

            candidate_r = np.asarray(residual(candidate), dtype=float)
            candidate_norm = float(norm(candidate_r))
            if candidate_norm < current_residual_norm:
                return _StepResult(
                    x=candidate,
                    increment=candidate_dx,
                    converged=True,
                    message="Line-search step accepted.",
                )

            alpha *= 0.5

        if (
            cfg.relaxed_residual_tol is not None
            and current_residual_norm <= cfg.relaxed_residual_tol
        ):
            return _StepResult(
                x=x,
                increment=np.zeros_like(dx),
                converged=True,
                accepted_solution=True,
                message="Accepted stalled near-converged iterate.",
            )

        return _StepResult(
            x=x,
            increment=np.zeros_like(dx),
            converged=False,
            message=(
                "Line search failed to reduce residual from "
                f"{current_residual_norm:.6e} at iteration {iteration}."
            ),
        )


@dataclass(frozen=True)
class _StepResult:
    x: np.ndarray
    increment: np.ndarray
    converged: bool
    message: str
    accepted_solution: bool = False


def _euclidean_norm(residual: np.ndarray) -> float:
    return float(np.linalg.norm(residual, ord=2))
