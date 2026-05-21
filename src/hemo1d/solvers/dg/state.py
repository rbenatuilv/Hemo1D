from __future__ import annotations

import numpy as np


class DGState:
    """
    Polynomial DG state for one vessel.

    Storage convention
    ------------------
    A, Q shape:
        (num_cells, num_local_dofs)

    degree = 0:
        one coefficient per cell, interpreted as the cell value.

    degree = 1:
        nodal representation on the reference element xi in [-1, 1]:

            dof 0 = left trace value inside the cell
            dof 1 = right trace value inside the cell

        The cell average is therefore:

            U_bar = 0.5 * (U_left + U_right)

        and the slope mode is:

            U_slope = 0.5 * (U_right - U_left)

    This representation is intentionally simple because endpoint traces,
    interface fluxes, and slope limiting are all direct array operations.
    """

    def __init__(self, num_cells: int, degree: int, name: str = "") -> None:
        if num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if degree not in (0, 1):
            raise NotImplementedError(
                "The array-based DG implementation currently supports degree 0 and 1."
            )

        self.num_cells = int(num_cells)
        self.degree = int(degree)
        self.num_local_dofs = self.degree + 1
        self.name = name

        self.A = np.empty((self.num_cells, self.num_local_dofs), dtype=np.float64)
        self.Q = np.empty((self.num_cells, self.num_local_dofs), dtype=np.float64)

    def scatter_forward(self) -> None:
        """
        Compatibility method with CGState.

        DGState is serial and array-based, so there are no ghost values to sync.
        """
        return None

    def copy_from(self, other: DGState) -> None:
        """
        Copy another DG state into this one.
        """
        self._check_compatible(other)

        self.A[:, :] = other.A
        self.Q[:, :] = other.Q
        self.scatter_forward()

    def cell_average_A(self) -> np.ndarray:
        """
        Return cell averages of A.

        For DG0 this is A[:, 0].
        For DG1 this is 0.5 * (A_left + A_right).
        """
        return self.cell_average(self.A)

    def cell_average_Q(self) -> np.ndarray:
        """
        Return cell averages of Q.

        For DG0 this is Q[:, 0].
        For DG1 this is 0.5 * (Q_left + Q_right).
        """
        return self.cell_average(self.Q)

    def cell_average(self, values: np.ndarray) -> np.ndarray:
        """
        Return cell averages of a scalar DG field with this state's layout.
        """
        self._check_scalar_field_shape(values)

        if self.degree == 0:
            return values[:, 0].copy()

        return 0.5 * (values[:, 0] + values[:, 1])

    def set_cell_average(self, values: np.ndarray, averages: np.ndarray) -> None:
        """
        Set a scalar DG field to piecewise constants with given cell averages.
        """
        self._check_scalar_field_shape(values)

        averages = np.asarray(averages, dtype=np.float64)
        if averages.shape != (self.num_cells,):
            raise ValueError(
                f"averages must have shape ({self.num_cells},), got {averages.shape}."
            )

        values[:, :] = averages[:, None]

    def assert_finite(self) -> None:
        """
        Raise if A or Q contains NaN/Inf.
        """
        if not np.all(np.isfinite(self.A)):
            raise RuntimeError(f"Non-finite area encountered in DG state '{self.name}'.")
        if not np.all(np.isfinite(self.Q)):
            raise RuntimeError(
                f"Non-finite flow rate encountered in DG state '{self.name}'."
            )

    def assert_positive_area(self, floor: float = 0.0) -> None:
        """
        Raise if any nodal area is <= floor.
        """
        self.assert_finite()

        min_A = float(np.min(self.A))
        if min_A <= floor:
            cell, dof = np.unravel_index(np.argmin(self.A), self.A.shape)
            raise RuntimeError(
                "Non-positive area encountered in DG state "
                "(area is non-positive or below limiter floor): "
                f"state={self.name!r}, cell={cell}, dof={dof}, "
                f"A_min={min_A:.16e}, floor={floor:.16e}."
            )

    def _check_compatible(self, other: DGState) -> None:
        if self.num_cells != other.num_cells:
            raise ValueError(
                f"Cannot copy DGState with {other.num_cells} cells into "
                f"DGState with {self.num_cells} cells."
            )
        if self.degree != other.degree:
            raise ValueError(
                f"Cannot copy DGState with degree {other.degree} into "
                f"DGState with degree {self.degree}."
            )

    def _check_scalar_field_shape(self, values: np.ndarray) -> None:
        if values.shape != (self.num_cells, self.num_local_dofs):
            raise ValueError(
                "Scalar DG field has incompatible shape: "
                f"expected {(self.num_cells, self.num_local_dofs)}, got {values.shape}."
            )
