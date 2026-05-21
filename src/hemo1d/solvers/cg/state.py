from __future__ import annotations

from dolfinx import fem


class CGState:
    """
    Discrete CG state for one vessel.

    We store A and Q as separate scalar CG functions.

    This is more convenient than a vector-valued function because:
        - endpoint area and flow are accessed separately;
        - boundary conditions often prescribe either A or Q;
        - junction solvers need scalar endpoint values.
    """

    

    def __init__(self, V: fem.FunctionSpace, name: str = "") -> None:
        self.A = fem.Function(V)
        self.Q = fem.Function(V)

        prefix = f"{name}_" if name else ""

        self.A.name = f"{prefix}A"
        self.Q.name = f"{prefix}Q"

    def scatter_forward(self) -> None:
        """
        Synchronize ghost values.

        Harmless in serial. Needed later for MPI runs.
        """
        self.A.x.scatter_forward()
        self.Q.x.scatter_forward()

    def copy_from(self, other: CGState) -> None:
        """
        Copy another CG state into this one.
        """
        self.A.x.array[:] = other.A.x.array
        self.Q.x.array[:] = other.Q.x.array
        self.scatter_forward()