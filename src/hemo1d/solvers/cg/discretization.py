from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import BoundaryState, EndpointData, EndpointSide, StateAtPoint
from hemo1d.solvers.base import VesselDiscretization
from hemo1d.solvers.cg.state import CGState


@dataclass(frozen=True)
class CGMeshConfig:
    """
    Mesh configuration for a single straight 1D vessel.

    length:
        Vessel length in cm.

    num_cells:
        Number of interval cells.

    degree:
        Polynomial degree of the continuous Lagrange space.

    For the initial Taylor-Galerkin implementation we use degree = 1, matching
    the linear finite elements described in the thesis.
    """

    length: float
    num_cells: int
    degree: int = 1

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("length must be positive.")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if self.degree <= 0:
            raise ValueError("degree must be positive.")


class CGFEMDiscretization(VesselDiscretization):
    """
    Continuous Galerkin FEM discretization for one 1D vessel.

    This class owns:
        - interval mesh [0, L]
        - scalar CG function space
        - endpoint dof lookup
        - cached sorted DOF coordinates (for performance)

    It does not own:
        - physics formulas
        - time stepping
        - boundary condition logic
    """

    def __init__(
        self,
        config: CGMeshConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self.config = config
        self.comm = comm

        self.domain = mesh.create_interval(
            comm,
            config.num_cells,
            [0.0, config.length],
        )

        self.V = fem.functionspace(
            self.domain,
            ("Lagrange", config.degree),
        )

        # Cache sorted DOF coordinates for efficient sampling
        # This avoids O(n log n) sorting on every probe/snapshot sample
        self._cached_sorted_dofs: np.ndarray | None = None
        self._cached_sorted_coords: np.ndarray | None = None

    @property
    def length(self) -> float:
        return self.config.length

    @property
    def num_cells(self) -> int:
        return self.config.num_cells

    @property
    def degree(self) -> int:
        return self.config.degree

    def create_state(self, name: str = "") -> CGState:
        return CGState(self.V, name=name)

    def interpolate_rest_state(self, state: CGState, physics: Hemo1DPhysics) -> None:
        """
        Set:
            A(z) = A0
            Q(z) = 0
        """
        A0 = physics.params.area0

        state.A.interpolate(lambda x: np.full(x.shape[1], A0, dtype=np.float64))  # type: ignore[attr-defined]
        state.Q.interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))  # type: ignore[attr-defined]

        state.scatter_forward()

    def coordinates(self) -> np.ndarray:
        """
        Return local mesh coordinates.

        In serial this is the full coordinate array.
        In parallel this is rank-local geometry data.
        """
        return self.domain.geometry.x[:, 0].copy()

    def dof_coordinates_sorted(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return DOF indices and coordinates sorted by spatial position.

        This is the primary interface for sampling operations. Results are cached
        to avoid expensive sorting on each call.

        Returns:
            (dof_indices, coordinates): Both sorted by increasing spatial position.
        """
        return self._sorted_dofs_and_coordinates()

    def h_min(self) -> float:
        """
        Minimum mesh spacing.

        For a uniform serial interval mesh this is length / num_cells.
        """
        x = np.sort(self.coordinates())
        dx = np.diff(x)

        if dx.size == 0:
            local_h = np.inf
        else:
            local_h = float(np.min(dx))

        return float(self.comm.allreduce(local_h, op=MPI.MIN))

    def num_dofs(self) -> int:
        """
        Number of scalar local dofs.
        """
        return self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs

    def locate_left_dofs(self) -> np.ndarray:
        return fem.locate_dofs_geometrical(
            self.V,
            lambda x: np.isclose(x[0], 0.0),
        )

    def locate_right_dofs(self) -> np.ndarray:
        L = self.length
        return fem.locate_dofs_geometrical(
            self.V,
            lambda x: np.isclose(x[0], L),
        )

    def endpoint_dofs(self, side: EndpointSide) -> np.ndarray:
        if side == EndpointSide.LEFT:
            return self.locate_left_dofs()
        if side == EndpointSide.RIGHT:
            return self.locate_right_dofs()
        raise ValueError(f"Unknown endpoint side: {side}")

    def _single_endpoint_dof(self, side: EndpointSide) -> int:
        """
        Return endpoint dof index.

        For now this is serial-oriented. Later we will make endpoint ownership
        MPI-safe for network simulations.
        """
        dofs = self.endpoint_dofs(side)

        if len(dofs) != 1:
            raise RuntimeError(
                f"Expected exactly one endpoint dof for {side}, got {len(dofs)}. "
                "For now, run endpoint tests in serial."
            )

        return int(dofs[0])

    def endpoint_state(self, state: CGState, side: EndpointSide) -> StateAtPoint:
        """
        Extract A and Q at the endpoint.

        In CG this is simply the endpoint dof value.
        """
        dof = self._single_endpoint_dof(side)

        return StateAtPoint(
            area=float(state.A.x.array[dof]),  # type: ignore[attr-defined]
            flow_rate=float(state.Q.x.array[dof]),  # type: ignore[attr-defined]
        )

    def set_endpoint_state(
        self,
        state: CGState,
        side: EndpointSide,
        boundary_state: BoundaryState,
    ) -> None:
        """
        Strongly set endpoint values in a CG state.

        This is useful for tests and later for applying already-computed
        boundary/junction values.
        """
        dof = self._single_endpoint_dof(side)

        state.A.x.array[dof] = boundary_state.area  # type: ignore[attr-defined]
        state.Q.x.array[dof] = boundary_state.flow_rate  # type: ignore[attr-defined]

        state.scatter_forward()

    def _sorted_dofs_and_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return scalar dofs sorted by coordinate.

        Results are cached to avoid expensive O(n log n) sorting on each call.
        This is critical for performance when sampling many probes per timestep.

        Serial-oriented helper. This is enough for the current single-vessel
        implementation. Later we will make endpoint ownership MPI-safe.
        """
        # Lazy initialization of cache
        if self._cached_sorted_dofs is None:
            coords = self.V.tabulate_dof_coordinates()[:, 0]
            dofs = np.arange(len(coords), dtype=np.int32)
            order = np.argsort(coords)

            self._cached_sorted_dofs = dofs[order].copy()
            self._cached_sorted_coords = coords[order].copy()

        # At this point the caches are guaranteed to be initialized
        assert self._cached_sorted_dofs is not None and self._cached_sorted_coords is not None

        return self._cached_sorted_dofs, self._cached_sorted_coords

    def endpoint_derivatives(self, state: CGState, side: EndpointSide) -> tuple[float, float]:
        """
        Approximate dA/dz and dQ/dz at an endpoint.

        For degree-1 CG on a uniform interval, this is the derivative on the
        first or last cell.

        LEFT:
            derivative from first two dofs.

        RIGHT:
            derivative from last two dofs.
        """
        if self.degree != 1:
            raise NotImplementedError(
                "Endpoint derivative currently implemented only for degree=1."
            )

        dofs, coords = self._sorted_dofs_and_coordinates()

        if len(dofs) < 2:
            raise RuntimeError("Need at least two dofs to compute endpoint derivative.")

        if side == EndpointSide.LEFT:
            i0 = int(dofs[0])
            i1 = int(dofs[1])
            z0 = float(coords[0])
            z1 = float(coords[1])
        elif side == EndpointSide.RIGHT:
            i0 = int(dofs[-2])
            i1 = int(dofs[-1])
            z0 = float(coords[-2])
            z1 = float(coords[-1])
        else:
            raise ValueError(f"Unknown endpoint side: {side}")

        dz = z1 - z0
        if dz <= 0.0:
            raise RuntimeError("Invalid coordinate ordering for endpoint derivative.")

        dA_dz = (float(state.A.x.array[i1]) - float(state.A.x.array[i0])) / dz  # type: ignore[attr-defined]
        dQ_dz = (float(state.Q.x.array[i1]) - float(state.Q.x.array[i0])) / dz  # type: ignore[attr-defined]

        return dA_dz, dQ_dz

    def endpoint_data(self, state: CGState, side: EndpointSide) -> EndpointData:
        """
        Return state and spatial derivative at an endpoint.
        """
        point_state = self.endpoint_state(state, side)
        dA_dz, dQ_dz = self.endpoint_derivatives(state, side)

        return EndpointData(
            state=point_state,
            d_area_dz=dA_dz,
            d_flow_rate_dz=dQ_dz,
        )