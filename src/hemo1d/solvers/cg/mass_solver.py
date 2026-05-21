from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg.discretization import CGFEMDiscretization


class CGScalarMassSolver:
    """
    Solver for scalar CG mass-matrix systems.

    It solves:

        M u = b

    where:

        M_ij = integral phi_i phi_j dz

    Endpoint values can be imposed strongly through Dirichlet conditions.

    This is used by the Taylor-Galerkin stepper to update A and Q separately.
    """

    def __init__(self, discretization: CGFEMDiscretization) -> None:
        self.discretization = discretization
        self.V = discretization.V

        self.left_dofs = discretization.endpoint_dofs(EndpointSide.LEFT)
        self.right_dofs = discretization.endpoint_dofs(EndpointSide.RIGHT)

        if len(self.left_dofs) != 1 or len(self.right_dofs) != 1:
            raise RuntimeError(
                "Expected exactly one left and one right endpoint dof. "
                "For now, use this solver in serial."
            )

        trial = ufl.TrialFunction(self.V)
        test = ufl.TestFunction(self.V)
        dx = ufl.dx(domain=discretization.domain)

        self.mass_form = fem.form(trial * test * dx)

        # Matrix is assembled with zero-valued endpoint Dirichlet BCs.
        # The actual endpoint values are applied to the RHS during solve().
        zero_left = fem.dirichletbc(
            PETSc.ScalarType(0.0),
            self.left_dofs,
            self.V,
        )
        zero_right = fem.dirichletbc(
            PETSc.ScalarType(0.0),
            self.right_dofs,
            self.V,
        )

        self._matrix_bcs = [zero_left, zero_right]

        self.matrix = fem_petsc.assemble_matrix(
            self.mass_form,
            bcs=self._matrix_bcs,
        )
        self.matrix.assemble()

        self.ksp = PETSc.KSP().create(discretization.comm)
        self.ksp.setOperators(self.matrix)
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        self.ksp.setTolerances(
            rtol=1.0e-12,
            atol=1.0e-14,
            max_it=500,
        )
        self.ksp.setFromOptions()

    def solve(
        self,
        rhs_form: fem.Form,
        out: fem.Function,
        left_value: float,
        right_value: float,
    ) -> None:
        """
        Assemble rhs_form and solve for out.

        Parameters
        ----------
        rhs_form:
            Linear form defining the RHS vector.

        out:
            Output function.

        left_value:
            Strong value at z = 0.

        right_value:
            Strong value at z = L.
        """
        left_bc = fem.dirichletbc(
            PETSc.ScalarType(left_value),
            self.left_dofs,
            self.V,
        )
        right_bc = fem.dirichletbc(
            PETSc.ScalarType(right_value),
            self.right_dofs,
            self.V,
        )
        bcs = [left_bc, right_bc]

        rhs = fem_petsc.assemble_vector(rhs_form)
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        fem_petsc.apply_lifting(
            rhs,
            [self.mass_form],
            [bcs],
        )
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        fem_petsc.set_bc(rhs, bcs)

        self.ksp.solve(rhs, out.x.petsc_vec)
        out.x.scatter_forward()

        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Mass solve failed with PETSc reason {reason}.")