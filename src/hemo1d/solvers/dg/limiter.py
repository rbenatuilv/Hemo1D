from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hemo1d.solvers.dg.state import DGState


@dataclass(frozen=True)
class DGLimiterConfig:
    """
    Configuration for DG slope/positivity limiting.

    enabled:
        Master switch.

    slope:
        Apply minmod slope limiting to DG1.

    positivity:
        Apply Zhang-Shu-style linear rescaling to keep nodal area above floor.

    area_floor:
        Absolute minimum allowed area.

        For your physiological tests, a reasonable value is often something like:

            area_floor = 1e-10

        or a small fraction of the vessel A0, which the stepper can pass later.

    minmod_beta:
        Generalized minmod parameter.

        beta = 1.0:
            classic TVD minmod, more dissipative.

        beta in [1.5, 2.0]:
            less dissipative, often better for smooth waves.

        For debugging DG1 stability, start with beta = 1.0.

    limit_area:
        Apply slope limiter to A.

    limit_flow_rate:
        Apply slope limiter to Q.

    raise_on_bad_average:
        If True, raise when a cell average of A is already <= area_floor.
        A slope limiter cannot repair a bad cell average.
    """

    enabled: bool = True
    slope: bool = True
    positivity: bool = True
    area_floor: float = 1.0e-12
    minmod_beta: float = 1.0
    limit_area: bool = True
    limit_flow_rate: bool = True
    raise_on_bad_average: bool = True

    def __post_init__(self) -> None:
        if self.area_floor < 0.0:
            raise ValueError("area_floor must be non-negative.")
        if self.minmod_beta <= 0.0:
            raise ValueError("minmod_beta must be positive.")


@dataclass(frozen=True)
class DGLimiterStats:
    """
    Small diagnostic object returned by DGSlopeLimiter.apply().
    """

    slope_limited_cells_A: int = 0
    slope_limited_cells_Q: int = 0
    positivity_limited_cells_A: int = 0
    min_area_before: float = np.inf
    min_area_after: float = np.inf
    min_area_average: float = np.inf


class DGSlopeLimiter:
    """
    DG0/DG1 slope and positivity limiter.

    For degree 0:
        no slope exists, so only sanity checks are applied.

    For degree 1:
        each cell stores nodal values:

            U_L = U[e, 0]
            U_R = U[e, 1]

        The cell average and slope mode are:

            U_bar = 0.5 * (U_L + U_R)
            delta = 0.5 * (U_R - U_L)

        The limiter replaces delta by a minmod-limited delta, then reconstructs:

            U_L <- U_bar - delta_limited
            U_R <- U_bar + delta_limited

    This preserves cell averages exactly.
    """

    def __init__(self, config: DGLimiterConfig | None = None) -> None:
        self.config = config or DGLimiterConfig()

    def apply(self, state: DGState) -> DGLimiterStats:
        """
        Limit a DG state in-place.
        """
        if not self.config.enabled:
            state.assert_finite()
            return DGLimiterStats(
                min_area_before=float(np.min(state.A)),
                min_area_after=float(np.min(state.A)),
                min_area_average=float(np.min(state.cell_average_A())),
            )

        state.assert_finite()

        min_area_before = float(np.min(state.A))
        min_area_average = float(np.min(state.cell_average_A()))

        if state.degree == 0:
            self._check_cell_average_area(state)
            state.assert_positive_area(self.config.area_floor)
            return DGLimiterStats(
                min_area_before=min_area_before,
                min_area_after=float(np.min(state.A)),
                min_area_average=min_area_average,
            )

        if state.degree != 1:
            raise NotImplementedError("DGSlopeLimiter currently supports DG0 and DG1 only.")

        slope_limited_A = 0
        slope_limited_Q = 0
        positivity_limited_A = 0

        if self.config.slope:
            if self.config.limit_area:
                slope_limited_A = self._limit_scalar_slopes(state.A)

            if self.config.limit_flow_rate:
                slope_limited_Q = self._limit_scalar_slopes(state.Q)

        if self.config.positivity:
            positivity_limited_A = self._apply_area_positivity(state)

        state.scatter_forward()

        min_area_after = float(np.min(state.A))

        return DGLimiterStats(
            slope_limited_cells_A=slope_limited_A,
            slope_limited_cells_Q=slope_limited_Q,
            positivity_limited_cells_A=positivity_limited_A,
            min_area_before=min_area_before,
            min_area_after=min_area_after,
            min_area_average=min_area_average,
        )

    def _limit_scalar_slopes(self, values: np.ndarray) -> int:
        """
        Minmod-limit DG1 slopes of a scalar field in-place.

        Boundary cells:
            Use one-sided availability. If a neighbor is missing, do not invent
            a boundary extrapolation here. Boundary states enter through the
            numerical flux. For limiting, missing side falls back to the current
            slope bound, which avoids forcing boundary cells to constants.
        """
        num_cells = values.shape[0]

        averages = 0.5 * (values[:, 0] + values[:, 1])
        current_delta = 0.5 * (values[:, 1] - values[:, 0])

        left_delta = np.empty_like(averages)
        right_delta = np.empty_like(averages)

        left_delta[0] = current_delta[0]
        left_delta[1:] = averages[1:] - averages[:-1]

        right_delta[:-1] = averages[1:] - averages[:-1]
        right_delta[-1] = current_delta[-1]

        beta = self.config.minmod_beta

        limited_delta = _minmod_many(
            current_delta,
            beta * left_delta,
            beta * right_delta,
        )

        changed = np.abs(limited_delta - current_delta) > 1.0e-14 * (
            1.0 + np.abs(current_delta)
        )

        values[:, 0] = averages - limited_delta
        values[:, 1] = averages + limited_delta

        return int(np.count_nonzero(changed))

    def _apply_area_positivity(self, state: DGState) -> int:
        """
        Rescale the A slope in each cell so nodal A >= area_floor.

        This preserves the cell average.

        If the cell average is already <= area_floor, no slope limiter can fix it.
        In that case the correct action is to reduce dt/CFL or fix boundary/junction
        data, so we raise a clear error by default.
        """
        floor = self.config.area_floor
        averages = 0.5 * (state.A[:, 0] + state.A[:, 1])

        bad_average = averages <= floor
        if np.any(bad_average):
            cell = int(np.where(bad_average)[0][0])
            msg = (
                "DG area limiter cannot repair a non-positive/small cell average: "
                f"cell={cell}, A_bar={averages[cell]:.16e}, "
                f"A_min={np.min(state.A[cell, :]):.16e}, floor={floor:.16e}. "
                "Reduce dt/CFL or fix boundary/junction data."
            )
            if self.config.raise_on_bad_average:
                raise RuntimeError(msg)

        theta = np.ones_like(averages)

        for dof in range(state.num_local_dofs):
            A_node = state.A[:, dof]
            below = A_node < floor

            denom = averages[below] - A_node[below]
            valid = denom > 0.0

            local_theta = np.ones(np.count_nonzero(below), dtype=np.float64)
            local_theta[valid] = (averages[below][valid] - floor) / denom[valid]
            local_theta = np.clip(local_theta, 0.0, 1.0)

            theta[below] = np.minimum(theta[below], local_theta)

        limited = theta < 1.0

        state.A[:, 0] = averages + theta * (state.A[:, 0] - averages)
        state.A[:, 1] = averages + theta * (state.A[:, 1] - averages)

        return int(np.count_nonzero(limited))

    def _check_cell_average_area(self, state: DGState) -> None:
        floor = self.config.area_floor
        averages = state.cell_average_A()

        bad = averages <= floor
        if np.any(bad):
            cell = int(np.where(bad)[0][0])
            msg = (
                "DG state has non-positive/small cell-average area: "
                f"cell={cell}, A_bar={averages[cell]:.16e}, floor={floor:.16e}."
            )
            if self.config.raise_on_bad_average:
                raise RuntimeError(msg)


def _minmod_many(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized three-argument minmod.

    Returns the argument with smallest absolute value if all signs agree.
    Otherwise returns 0.
    """
    same_positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    same_negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    same_sign = same_positive | same_negative

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    abs_c = np.abs(c)

    min_abs = np.minimum(abs_a, np.minimum(abs_b, abs_c))

    sign = np.sign(a)

    out = np.zeros_like(a)
    out[same_sign] = sign[same_sign] * min_abs[same_sign]

    return out