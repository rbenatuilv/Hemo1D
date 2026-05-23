from __future__ import annotations

import numpy as np

from hemo1d.boundary.junction.data import (
    JunctionData,
    JunctionSolution,
)
from hemo1d.boundary.junction.residual import JunctionResidual
from hemo1d.core.newton import NewtonConfig, NewtonSolver
from hemo1d.core.state import BoundaryState


class JunctionSolver:
    """Newton-based solver for one two- or three-vessel junction."""

    def __init__(
        self,
        newton_solver: NewtonSolver | None = None,
    ) -> None:
        self.newton_solver = (
            newton_solver
            if newton_solver is not None
            else NewtonSolver(
                NewtonConfig(
                    residual_tol=1.0e-08,
                    increment_tol=1.0e-08,
                    max_iterations=20,
                )
            )
        )

    def solve(
        self,
        data: JunctionData,
        dt: float,
        x0: np.ndarray | None = None,
        raise_on_failure: bool = True,
    ) -> JunctionSolution:
        residual = JunctionResidual(
            data=data,
            dt=dt,
        )

        if x0 is None:
            x0 = residual.initial_guess()

        result = self.newton_solver.solve(
            residual=residual,
            x0=x0,
            jacobian=residual.jacobian,
            raise_on_failure=raise_on_failure,
        )

        endpoint_states = tuple(
            BoundaryState(
                area=float(result.x[2 * i]),
                flow_rate=float(result.x[2 * i + 1]),
            )
            for i in range(len(data.endpoints))
        )

        return JunctionSolution(
            endpoint_states=endpoint_states,
            newton_result=result,
        )


__all__ = ["JunctionSolver"]
