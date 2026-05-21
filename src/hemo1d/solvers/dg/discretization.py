from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide, StateAtPoint
from hemo1d.solvers.base import VesselDiscretization
from hemo1d.solvers.dg.state import DGState


ScalarFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DGMeshConfig:
    """
    Mesh configuration for one straight 1D DG vessel.

    length:
        Vessel length in cm.

    num_cells:
        Number of DG cells.

    degree:
        Local polynomial degree.

    Current implementation:
        degree = 0 or degree = 1.

    DG1 convention:
        local nodal basis on xi = [-1, 1].
    """

    length: float
    num_cells: int
    degree: int = 0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("length must be positive.")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if self.degree not in (0, 1):
            raise NotImplementedError(
                "The array-based DG implementation currently supports degree 0 and 1."
            )


class DGFEMDiscretization(VesselDiscretization):
    """
    Array-based DG discretization for one 1D vessel.

    This class owns only the geometric and polynomial data:

        - uniform mesh on [0, L]
        - reference basis values
        - quadrature points/weights
        - local mass matrix and inverse

    It does not own:

        - physics formulas
        - time stepping
        - limiter
        - boundary/junction logic

    Reference mapping
    -----------------
    xi in [-1, 1]

        x = x_center[e] + (h / 2) xi
        dx = (h / 2) dxi
        d/dx = (2 / h) d/dxi
    """

    def __init__(self, config: DGMeshConfig) -> None:
        self.config = config

        self._h = config.length / config.num_cells
        self._cell_edges = np.linspace(
            0.0,
            config.length,
            config.num_cells + 1,
            dtype=np.float64,
        )
        self._cell_centers = 0.5 * (self._cell_edges[:-1] + self._cell_edges[1:])

        self.reference_nodes = self._build_reference_nodes(config.degree)
        self.quad_points, self.quad_weights = self._build_quadrature(config.degree)

        self.basis_at_quad = self._eval_basis(self.quad_points)
        self.basis_derivative_at_quad = self._eval_basis_derivative(self.quad_points)

        self.basis_at_left = self._eval_basis(np.array([-1.0], dtype=np.float64))[0]
        self.basis_at_right = self._eval_basis(np.array([1.0], dtype=np.float64))[0]

        self.reference_mass_matrix = self._build_reference_mass_matrix()
        self.mass_matrix = 0.5 * self.h * self.reference_mass_matrix
        self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)

    @property
    def length(self) -> float:
        return self.config.length

    @property
    def num_cells(self) -> int:
        return self.config.num_cells

    @property
    def degree(self) -> int:
        return self.config.degree

    @property
    def num_local_dofs(self) -> int:
        return self.degree + 1

    @property
    def h(self) -> float:
        return self._h

    def h_min(self) -> float:
        return self._h

    def cell_edges(self) -> np.ndarray:
        return self._cell_edges.copy()

    def cell_centers(self) -> np.ndarray:
        return self._cell_centers.copy()

    def coordinates(self) -> np.ndarray:
        """
        Physical coordinates of all local DG nodes.

        Shape:
            (num_cells, num_local_dofs)
        """
        return self.physical_points(self.reference_nodes)

    def num_dofs(self) -> int:
        """
        Number of scalar DOFs per variable.
        """
        return self.num_cells * self.num_local_dofs

    def create_state(self, name: str = "") -> DGState:
        return DGState(
            num_cells=self.num_cells,
            degree=self.degree,
            name=name,
        )

    def interpolate_rest_state(self, state: DGState, physics: Hemo1DPhysics) -> None:
        """
        Set A(z) = A0 and Q(z) = 0.
        """
        self._check_state_compatible(state)

        state.A[:, :] = physics.params.area0
        state.Q[:, :] = 0.0
        state.scatter_forward()

    def interpolate_state(
        self,
        state: DGState,
        area_fn: ScalarFunction,
        flow_rate_fn: ScalarFunction,
    ) -> None:
        """
        Interpolate scalar functions into DG nodal values.

        area_fn and flow_rate_fn receive an array of physical coordinates with
        shape (num_cells, num_local_dofs).
        """
        self._check_state_compatible(state)

        x = self.coordinates()
        state.A[:, :] = area_fn(x)
        state.Q[:, :] = flow_rate_fn(x)
        state.scatter_forward()

    def physical_points(self, reference_points: np.ndarray) -> np.ndarray:
        """
        Map reference points xi to physical points in every cell.

        reference_points:
            shape (n_points,)

        returns:
            shape (num_cells, n_points)
        """
        xi = np.asarray(reference_points, dtype=np.float64)
        return self._cell_centers[:, None] + 0.5 * self.h * xi[None, :]

    def evaluate_on_reference_points(
        self,
        coeffs: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate a DG scalar field on reference points.

        coeffs may be either:

            (num_cells, num_local_dofs)

        or a subset of cells, for example:

            (1, num_local_dofs)

        This is needed by the DG sampling helpers, which evaluate only the cell
        containing the probe coordinate.
        """
        self._check_scalar_coeffs_local(coeffs)

        basis = self._eval_basis(np.asarray(reference_points, dtype=np.float64))
        return coeffs @ basis.T
    
    def _check_scalar_coeffs_local(self, coeffs: np.ndarray) -> None:
        """
        Check scalar DG coefficient array allowing any number of cells.

        Expected shape:
            (n_cells, num_local_dofs)

        where n_cells can be 1, num_cells, or any positive subset length.
        """
        if coeffs.ndim != 2:
            raise ValueError(
                f"Expected scalar coeffs to be 2D, got shape {coeffs.shape}."
            )

        if coeffs.shape[0] <= 0:
            raise ValueError("Expected at least one cell in scalar coeffs.")

        if coeffs.shape[1] != self.num_local_dofs:
            raise ValueError(
                "Expected scalar coeffs with "
                f"{self.num_local_dofs} local dofs, got shape {coeffs.shape}."
            )

    def evaluate_at_quadrature(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Evaluate scalar DG coefficients at quadrature points.

        returns:
            shape (num_cells, num_quad_points)
        """
        self._check_scalar_coeffs(coeffs)
        return coeffs @ self.basis_at_quad.T

    def trace_value(
        self,
        coeffs: np.ndarray,
        cell: int,
        side: EndpointSide,
    ) -> float:
        """
        Evaluate one scalar DG field at a cell trace.
        """
        self._check_scalar_coeffs(coeffs)

        if cell < 0 or cell >= self.num_cells:
            raise IndexError(f"cell index out of range: {cell}")

        basis = self.trace_basis(side)
        return float(np.dot(coeffs[cell, :], basis))

    def trace_basis(self, side: EndpointSide) -> np.ndarray:
        """
        Basis vector evaluated at a cell trace.
        """
        if side == EndpointSide.LEFT:
            return self.basis_at_left
        if side == EndpointSide.RIGHT:
            return self.basis_at_right
        raise ValueError(f"Unknown endpoint side: {side}")

    def endpoint_state(self, state: DGState, side: EndpointSide) -> StateAtPoint:
        """
        Extract the interior DG trace at a vessel endpoint.
        """
        self._check_state_compatible(state)

        cell = self._endpoint_cell(side)

        return StateAtPoint(
            area=self.trace_value(state.A, cell, side),
            flow_rate=self.trace_value(state.Q, cell, side),
        )

    def endpoint_derivatives(self, state: DGState, side: EndpointSide) -> tuple[float, float]:
        """
        Compute dA/dz and dQ/dz from the endpoint-cell polynomial.

        DG0:
            derivative is zero.

        DG1:
            derivative is constant in the cell.
        """
        self._check_state_compatible(state)

        cell = self._endpoint_cell(side)
        xi = np.array([-1.0 if side == EndpointSide.LEFT else 1.0], dtype=np.float64)

        dphi_dxi = self._eval_basis_derivative(xi)[0]
        scale = 2.0 / self.h

        dA_dz = scale * float(np.dot(state.A[cell, :], dphi_dxi))
        dQ_dz = scale * float(np.dot(state.Q[cell, :], dphi_dxi))

        return dA_dz, dQ_dz

    def endpoint_data(self, state: DGState, side: EndpointSide) -> EndpointData:
        point_state = self.endpoint_state(state, side)
        dA_dz, dQ_dz = self.endpoint_derivatives(state, side)

        return EndpointData(
            state=point_state,
            d_area_dz=dA_dz,
            d_flow_rate_dz=dQ_dz,
        )

    def set_endpoint_state(
        self,
        state: DGState,
        side: EndpointSide,
        boundary_state: BoundaryState,
    ) -> None:
        """
        Strong endpoint-cell overwrite.

        Keep this only for simple tests/debugging. The real DG stepper should use
        BoundaryState as an exterior numerical-flux trace, not strongly impose it.
        """
        self._check_state_compatible(state)

        cell = self._endpoint_cell(side)

        state.A[cell, :] = boundary_state.area
        state.Q[cell, :] = boundary_state.flow_rate
        state.scatter_forward()

    def _endpoint_cell(self, side: EndpointSide) -> int:
        if side == EndpointSide.LEFT:
            return 0
        if side == EndpointSide.RIGHT:
            return self.num_cells - 1
        raise ValueError(f"Unknown endpoint side: {side}")

    def _build_reference_nodes(self, degree: int) -> np.ndarray:
        if degree == 0:
            return np.array([0.0], dtype=np.float64)
        if degree == 1:
            return np.array([-1.0, 1.0], dtype=np.float64)
        raise NotImplementedError

    def _build_quadrature(self, degree: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Legendre quadrature on [-1, 1].

        DG0:
            one point.

        DG1:
            three points. This is a robust default for nonlinear fluxes.
        """
        if degree == 0:
            return (
                np.array([0.0], dtype=np.float64),
                np.array([2.0], dtype=np.float64),
            )

        if degree == 1:
            return (
                np.array(
                    [
                        -np.sqrt(3.0 / 5.0),
                        0.0,
                        np.sqrt(3.0 / 5.0),
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        5.0 / 9.0,
                        8.0 / 9.0,
                        5.0 / 9.0,
                    ],
                    dtype=np.float64,
                ),
            )

        raise NotImplementedError

    def _eval_basis(self, xi: np.ndarray) -> np.ndarray:
        """
        Evaluate nodal Lagrange basis functions.

        returns:
            basis[q, m] = phi_m(xi_q)
        """
        xi = np.asarray(xi, dtype=np.float64)

        if self.degree == 0:
            return np.ones((xi.size, 1), dtype=np.float64)

        if self.degree == 1:
            phi_left = 0.5 * (1.0 - xi)
            phi_right = 0.5 * (1.0 + xi)
            return np.column_stack([phi_left, phi_right])

        raise NotImplementedError

    def _eval_basis_derivative(self, xi: np.ndarray) -> np.ndarray:
        """
        Evaluate d(phi_m)/d(xi).

        returns:
            derivative[q, m] = dphi_m/dxi(xi_q)
        """
        xi = np.asarray(xi, dtype=np.float64)

        if self.degree == 0:
            return np.zeros((xi.size, 1), dtype=np.float64)

        if self.degree == 1:
            dphi_left = -0.5 * np.ones_like(xi)
            dphi_right = 0.5 * np.ones_like(xi)
            return np.column_stack([dphi_left, dphi_right])

        raise NotImplementedError

    def _build_reference_mass_matrix(self) -> np.ndarray:
        """
        M_ref[m, n] = integral_{-1}^{1} phi_m phi_n dxi.
        """
        if self.degree == 0:
            return np.array([[2.0]], dtype=np.float64)

        if self.degree == 1:
            return np.array(
                [
                    [2.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 2.0 / 3.0],
                ],
                dtype=np.float64,
            )

        raise NotImplementedError

    def _check_state_compatible(self, state: DGState) -> None:
        if state.num_cells != self.num_cells:
            raise ValueError(
                f"State has {state.num_cells} cells, but discretization has "
                f"{self.num_cells} cells."
            )
        if state.degree != self.degree:
            raise ValueError(
                f"State has degree {state.degree}, but discretization has "
                f"degree {self.degree}."
            )

    def _check_scalar_coeffs(self, coeffs: np.ndarray) -> None:
        expected = (self.num_cells, self.num_local_dofs)
        if coeffs.shape != expected:
            raise ValueError(f"Expected scalar coeffs shape {expected}, got {coeffs.shape}.")