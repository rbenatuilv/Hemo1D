from __future__ import annotations

import numpy as np

from hemo1d.boundary.junction.characteristics import (
    _mass_sign,
    _state_vector,
    compatibility_target,
    outgoing_left_eigenvector,
)
from hemo1d.boundary.junction.data import JunctionData
from hemo1d.boundary.junction.losses import (
    _d_pressure_loss_term_gradient,
    _pressure_loss_term,
    _total_pressure_gradient,
)


class JunctionResidual:
    """
    Residual of a two- or three-vessel junction system.

    Unknown vector:

        x = [A0, Q0, A1, Q1, ...]

    Equations:

        R0 = sum_i sign_i Q_i

        R1..R(N-1) = Ptot_0 - Ptot_i - loss_i

        RN..R(2N-1) = l_i^T U_i - l_i^T CC_i
    """

    def __init__(
        self,
        data: JunctionData,
        dt: float,
        include_density_in_total_pressure: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.data = data
        self.dt = dt
        self.include_density_in_total_pressure = include_density_in_total_pressure
        self.num_endpoints = len(data.endpoints)
        self.num_unknowns = 2 * self.num_endpoints

        self._compatibility_targets = tuple(
            compatibility_target(
                endpoint.physics,
                endpoint.endpoint_data,
                dt,
            )
            for endpoint in data.endpoints
        )
        self._outgoing_left_eigenvectors = tuple(
            outgoing_left_eigenvector(
                endpoint.physics,
                endpoint.endpoint_data,
                endpoint.side,
            )
            for endpoint in data.endpoints
        )

    def initial_guess(self) -> np.ndarray:
        """Previous endpoint values are the natural Newton initial guess."""
        values: list[float] = []
        for endpoint in self.data.endpoints:
            state = endpoint.endpoint_data.state
            values.extend((state.area, state.flow_rate))
        return np.array(values, dtype=float)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        self._check_shape(x)

        areas = x[0::2]
        flow_rates = x[1::2]
        if np.any(areas <= 0.0):
            # Newton may briefly try invalid states. Return a large residual
            # instead of evaluating sqrt on negative areas.
            return np.full(self.num_unknowns, 1.0e20, dtype=float)

        total_pressures = tuple(
            endpoint.physics.total_pressure(
                area,
                flow_rate,
                include_density=self.include_density_in_total_pressure,
            )
            for endpoint, area, flow_rate in zip(
                self.data.endpoints,
                areas,
                flow_rates,
                strict=True,
            )
        )
        state_vectors = tuple(
            _state_vector(area, flow_rate)
            for area, flow_rate in zip(areas, flow_rates, strict=True)
        )

        residual = np.zeros(self.num_unknowns, dtype=float)

        residual[0] = sum(
            _mass_sign(endpoint.side) * flow_rate
            for endpoint, flow_rate in zip(
                self.data.endpoints,
                flow_rates,
                strict=True,
            )
        )

        reference_pressure = total_pressures[0]
        for i in range(1, self.num_endpoints):
            endpoint = self.data.endpoints[i]
            residual[i] = (
                reference_pressure
                - total_pressures[i]
                - _pressure_loss_term(
                    endpoint.physics,
                    areas[i],
                    flow_rates[i],
                    endpoint.angle,
                )
            )

        for i, (left_eigenvector, state_vector, compatibility_target_value) in enumerate(
            zip(
                self._outgoing_left_eigenvectors,
                state_vectors,
                self._compatibility_targets,
                strict=True,
            )
        ):
            residual[self.num_endpoints + i] = float(
                left_eigenvector @ state_vector - left_eigenvector @ compatibility_target_value
            )

        return residual

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic Jacobian of the generic junction residual.

        Unknown vector:

            x = [A0, Q0, A1, Q1, ...]
        """
        x = np.asarray(x, dtype=float)

        self._check_shape(x)

        areas = x[0::2]
        flow_rates = x[1::2]
        if np.any(areas <= 0.0):
            # Do not build a fake Jacobian for an invalid physical state.
            # Let Newton fail clearly instead of producing misleading steps.
            labels = ", ".join(
                f"{endpoint.name}: A={area}"
                for endpoint, area in zip(self.data.endpoints, areas, strict=True)
                if area <= 0.0
            )
            raise ValueError(f"Cannot evaluate junction Jacobian with non-positive area: {labels}")

        total_pressure_gradients = tuple(
            _total_pressure_gradient(
                endpoint.physics,
                area,
                flow_rate,
                include_density=self.include_density_in_total_pressure,
            )
            for endpoint, area, flow_rate in zip(
                self.data.endpoints,
                areas,
                flow_rates,
                strict=True,
            )
        )

        J = np.zeros((self.num_unknowns, self.num_unknowns), dtype=float)

        for i, endpoint in enumerate(self.data.endpoints):
            # R0 = sum_i sign_i Q_i
            J[0, 2 * i + 1] = _mass_sign(endpoint.side)

        dptot_ref_dA, dptot_ref_dQ = total_pressure_gradients[0]
        for i in range(1, self.num_endpoints):
            endpoint = self.data.endpoints[i]
            row = i
            endpoint_col = 2 * i
            dptot_i_dA, dptot_i_dQ = total_pressure_gradients[i]
            dloss_dA, dloss_dQ = _d_pressure_loss_term_gradient(
                endpoint.physics,
                areas[i],
                flow_rates[i],
                endpoint.angle,
            )

            # Ri = Ptot_0 - Ptot_i - loss_i
            J[row, 0] = dptot_ref_dA
            J[row, 1] = dptot_ref_dQ
            J[row, endpoint_col] = -dptot_i_dA - dloss_dA
            J[row, endpoint_col + 1] = -dptot_i_dQ - dloss_dQ

        for i, left_eigenvector in enumerate(self._outgoing_left_eigenvectors):
            row = self.num_endpoints + i
            col = 2 * i
            J[row, col] = left_eigenvector[0]
            J[row, col + 1] = left_eigenvector[1]

        return J

    def _check_shape(self, x: np.ndarray) -> None:
        if x.shape != (self.num_unknowns,):
            raise ValueError(f"Junction unknown vector must have shape ({self.num_unknowns},).")


__all__ = ["JunctionResidual"]
