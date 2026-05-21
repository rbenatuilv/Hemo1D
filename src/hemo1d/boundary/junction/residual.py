from __future__ import annotations

import numpy as np

from hemo1d.boundary.junction.characteristics import (
    _mass_sign,
    _state_vector,
    compatibility_target,
    outgoing_left_eigenvector,
)
from hemo1d.boundary.junction.data import BifurcationJunctionData
from hemo1d.boundary.junction.losses import (
    _d_pressure_loss_term_gradient,
    _pressure_loss_term,
    _total_pressure_gradient,
)


class BifurcationJunctionResidual:
    """
    Residual of the 6-equation bifurcation system.

    Unknown vector:

        x = [A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2]

    Equations:

        R0 = Q_p - Q_d1 - Q_d2

        R1 = Ptot_p - Ptot_d1
        R2 = Ptot_p - Ptot_d2

        R3 = l_p^T U_p   - l_p^T CC_p
        R4 = l_d1^T U_d1 - l_d1^T CC_d1
        R5 = l_d2^T U_d2 - l_d2^T CC_d2

    This first version assumes total-pressure continuity, i.e. no angle/loss
    terms yet.
    """

    def __init__(
        self,
        data: BifurcationJunctionData,
        dt: float,
        include_density_in_total_pressure: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.data = data
        self.dt = dt
        self.include_density_in_total_pressure = include_density_in_total_pressure

        self._cc_parent = compatibility_target(
            data.parent.physics,
            data.parent.endpoint_data,
            dt,
        )
        self._cc_daughter1 = compatibility_target(
            data.daughter1.physics,
            data.daughter1.endpoint_data,
            dt,
        )
        self._cc_daughter2 = compatibility_target(
            data.daughter2.physics,
            data.daughter2.endpoint_data,
            dt,
        )

        self._l_parent = outgoing_left_eigenvector(
            data.parent.physics,
            data.parent.endpoint_data,
            data.parent.side,
        )
        self._l_daughter1 = outgoing_left_eigenvector(
            data.daughter1.physics,
            data.daughter1.endpoint_data,
            data.daughter1.side,
        )
        self._l_daughter2 = outgoing_left_eigenvector(
            data.daughter2.physics,
            data.daughter2.endpoint_data,
            data.daughter2.side,
        )

    def initial_guess(self) -> np.ndarray:
        """Previous endpoint values are the natural Newton initial guess."""
        p = self.data.parent.endpoint_data.state
        d1 = self.data.daughter1.endpoint_data.state
        d2 = self.data.daughter2.endpoint_data.state

        return np.array(
            [
                p.area,
                p.flow_rate,
                d1.area,
                d1.flow_rate,
                d2.area,
                d2.flow_rate,
            ],
            dtype=float,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if x.shape != (6,):
            raise ValueError("Bifurcation unknown vector must have shape (6,).")

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = x

        if A_p <= 0.0 or A_d1 <= 0.0 or A_d2 <= 0.0:
            # Newton may briefly try invalid states. Return a large residual
            # instead of evaluating sqrt on negative areas.
            return np.full(6, 1.0e20, dtype=float)

        physics_p = self.data.parent.physics
        physics_d1 = self.data.daughter1.physics
        physics_d2 = self.data.daughter2.physics

        ptot_p = physics_p.total_pressure(
            A_p,
            Q_p,
            include_density=self.include_density_in_total_pressure,
        )
        ptot_d1 = physics_d1.total_pressure(
            A_d1,
            Q_d1,
            include_density=self.include_density_in_total_pressure,
        )
        ptot_d2 = physics_d2.total_pressure(
            A_d2,
            Q_d2,
            include_density=self.include_density_in_total_pressure,
        )

        U_p = _state_vector(A_p, Q_p)
        U_d1 = _state_vector(A_d1, Q_d1)
        U_d2 = _state_vector(A_d2, Q_d2)

        residual = np.zeros(6, dtype=float)

        s1 = _mass_sign(self.data.parent.side)
        s2 = _mass_sign(self.data.daughter1.side)
        s3 = _mass_sign(self.data.daughter2.side)

        residual[0] = s1 * Q_p + s2 * Q_d1 + s3 * Q_d2
        residual[1] = ptot_p - ptot_d1 - _pressure_loss_term(
            physics_d1,
            A_d1,
            Q_d1,
            self.data.daughter1.angle,
        )
        residual[2] = ptot_p - ptot_d2 - _pressure_loss_term(
            physics_d2,
            A_d2,
            Q_d2,
            self.data.daughter2.angle,
        )

        residual[3] = float(self._l_parent @ U_p - self._l_parent @ self._cc_parent)
        residual[4] = float(
            self._l_daughter1 @ U_d1 - self._l_daughter1 @ self._cc_daughter1
        )
        residual[5] = float(
            self._l_daughter2 @ U_d2 - self._l_daughter2 @ self._cc_daughter2
        )

        return residual

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic Jacobian of the 6-equation bifurcation residual.

        Unknown vector:

            x = [A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2]
        """
        x = np.asarray(x, dtype=float)

        if x.shape != (6,):
            raise ValueError("Bifurcation unknown vector must have shape (6,).")

        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = x

        if A_p <= 0.0 or A_d1 <= 0.0 or A_d2 <= 0.0:
            # Do not build a fake Jacobian for an invalid physical state.
            # Let Newton fail clearly instead of producing misleading steps.
            raise ValueError(
                "Cannot evaluate bifurcation Jacobian with non-positive area: "
                f"A_p={A_p}, A_d1={A_d1}, A_d2={A_d2}"
            )

        physics_p = self.data.parent.physics
        physics_d1 = self.data.daughter1.physics
        physics_d2 = self.data.daughter2.physics

        dptot_p_dA, dptot_p_dQ = _total_pressure_gradient(
            physics_p,
            A_p,
            Q_p,
            include_density=self.include_density_in_total_pressure,
        )
        dptot_d1_dA, dptot_d1_dQ = _total_pressure_gradient(
            physics_d1,
            A_d1,
            Q_d1,
            include_density=self.include_density_in_total_pressure,
        )
        dptot_d2_dA, dptot_d2_dQ = _total_pressure_gradient(
            physics_d2,
            A_d2,
            Q_d2,
            include_density=self.include_density_in_total_pressure,
        )

        J = np.zeros((6, 6), dtype=float)

        s1 = _mass_sign(self.data.parent.side)
        s2 = _mass_sign(self.data.daughter1.side)
        s3 = _mass_sign(self.data.daughter2.side)

        dF2_d1 = _d_pressure_loss_term_gradient(
            physics_d1,
            A_d1,
            Q_d1,
            self.data.daughter1.angle,
        )
        dF_d2 = _d_pressure_loss_term_gradient(
            physics_d2,
            A_d2,
            Q_d2,
            self.data.daughter2.angle,
        )

        # R0 = s1 Q_p + s2 Q_d1 + s3 Q_d2
        J[0, 1] = s1
        J[0, 3] = s2
        J[0, 5] = s3

        # R1 = Ptot_p - Ptot_d1
        J[1, 0] = dptot_p_dA
        J[1, 1] = dptot_p_dQ
        J[1, 2] = -dptot_d1_dA - dF2_d1[0]
        J[1, 3] = -dptot_d1_dQ - dF2_d1[1]

        # R2 = Ptot_p - Ptot_d2
        J[2, 0] = dptot_p_dA
        J[2, 1] = dptot_p_dQ
        J[2, 4] = -dptot_d2_dA - dF_d2[0]
        J[2, 5] = -dptot_d2_dQ - dF_d2[1]

        # R3 = l_p^T U_p - const
        J[3, 0] = self._l_parent[0]
        J[3, 1] = self._l_parent[1]

        # R4 = l_d1^T U_d1 - const
        J[4, 2] = self._l_daughter1[0]
        J[4, 3] = self._l_daughter1[1]

        # R5 = l_d2^T U_d2 - const
        J[5, 4] = self._l_daughter2[0]
        J[5, 5] = self._l_daughter2[1]

        return J


__all__ = ["BifurcationJunctionResidual"]
