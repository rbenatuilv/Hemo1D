from __future__ import annotations

import numpy as np

from hemo1d.boundary.junction.data import BifurcationJunctionData, BifurcationSolution
from hemo1d.boundary.junction.residual import BifurcationJunctionResidual
from hemo1d.core.newton import NewtonConfig, NewtonSolver
from hemo1d.core.state import BoundaryState


class BifurcationJunctionSolver:
    """Newton-based solver for one 1-to-2 bifurcation."""

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
        data: BifurcationJunctionData,
        dt: float,
        x0: np.ndarray | None = None,
        raise_on_failure: bool = True,
    ) -> BifurcationSolution:
        residual = BifurcationJunctionResidual(
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

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = result.x

        return BifurcationSolution(
            parent=BoundaryState(area=float(A_p), flow_rate=float(Q_p)),
            daughter1=BoundaryState(area=float(A_d1), flow_rate=float(Q_d1)),
            daughter2=BoundaryState(area=float(A_d2), flow_rate=float(Q_d2)),
            newton_result=result,
        )


__all__ = ["BifurcationJunctionSolver"]
