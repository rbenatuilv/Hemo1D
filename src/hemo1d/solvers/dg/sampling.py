from __future__ import annotations

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import StateAtPoint
from hemo1d.solvers.dg.discretization import DGFEMDiscretization
from hemo1d.solvers.dg.state import DGState


def sample_dg_state(
    discretization: DGFEMDiscretization,
    state: DGState,
    physics: Hemo1DPhysics,
    coordinate: float,
) -> StateAtPoint:
    """
    Sample a DG state at one physical coordinate.

    For DG0:
        returns the cell constant.

    For DG1:
        evaluates the local linear polynomial inside the containing cell.

    DG is discontinuous at cell interfaces. If coordinate lies exactly at an
    interior interface, this function chooses the cell to the right, except at
    x = L where it chooses the last cell.
    """
    del physics

    cell, xi = _locate_cell_and_reference_coordinate(discretization, coordinate)
    reference_points = np.array([xi], dtype=np.float64)

    area = discretization.evaluate_on_reference_points(
        state.A[cell : cell + 1, :],
        reference_points,
    )[0, 0]
    flow_rate = discretization.evaluate_on_reference_points(
        state.Q[cell : cell + 1, :],
        reference_points,
    )[0, 0]

    return StateAtPoint(
        area=float(area),
        flow_rate=float(flow_rate),
    )


def sample_dg_state_array(
    discretization: DGFEMDiscretization,
    state: DGState,
    physics: Hemo1DPhysics,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a DG state at many physical coordinates.

    This vectorized path preserves the scalar sampling convention at
    discontinuities: an exact interior interface is sampled from the cell to
    the right, while x = L is sampled from the last cell.
    """
    del physics

    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 1:
        raise ValueError("coordinates must be one-dimensional.")
    if np.any(points < 0.0) or np.any(points > discretization.length):
        raise ValueError(
            f"Coordinates must lie inside domain [0, {discretization.length}]."
        )

    cells = np.floor(points / discretization.h).astype(np.int64)
    cells = np.clip(cells, 0, discretization.num_cells - 1)
    cells[np.isclose(points, discretization.length)] = discretization.num_cells - 1

    left_edges = cells * discretization.h
    xi = 2.0 * (points - left_edges) / discretization.h - 1.0
    xi = np.clip(xi, -1.0, 1.0)

    if discretization.degree == 0:
        return state.A[cells, 0].copy(), state.Q[cells, 0].copy()

    if discretization.degree == 1:
        phi_left = 0.5 * (1.0 - xi)
        phi_right = 0.5 * (1.0 + xi)
        areas = state.A[cells, 0] * phi_left + state.A[cells, 1] * phi_right
        flows = state.Q[cells, 0] * phi_left + state.Q[cells, 1] * phi_right
        return areas, flows

    raise NotImplementedError("DG sampling currently supports degree 0 and 1 only.")


def extract_dg_state_arrays(
    discretization: DGFEMDiscretization,
    state: DGState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract representative arrays from a DG state.

    Returns:
        z, A, Q

    DG0:
        one value per cell center.

    DG1:
        all local nodal values flattened in cell order.

    Important:
        DG nodes are discontinuous at interfaces. For DG1, an interior interface
        appears twice:

            - right trace of the left cell
            - left trace of the right cell

        This is intentional and useful for visualizing jumps.
    """
    z = discretization.coordinates()

    return (
        z.reshape(-1).copy(),
        state.A.reshape(-1).copy(),
        state.Q.reshape(-1).copy(),
    )


def _locate_cell_and_reference_coordinate(
    discretization: DGFEMDiscretization,
    coordinate: float,
) -> tuple[int, float]:
    """
    Locate the cell containing a physical coordinate and map it to xi in [-1, 1].
    """
    if coordinate < 0.0 or coordinate > discretization.length:
        raise ValueError(
            f"Coordinate {coordinate} is outside domain [0, {discretization.length}]."
        )

    if np.isclose(coordinate, discretization.length):
        cell = discretization.num_cells - 1
    else:
        cell = int(np.floor(coordinate / discretization.h))
        cell = min(max(cell, 0), discretization.num_cells - 1)

    left_edge = cell * discretization.h
    xi = 2.0 * (coordinate - left_edge) / discretization.h - 1.0

    xi = float(np.clip(xi, -1.0, 1.0))

    return cell, xi
