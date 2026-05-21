import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import EndpointSide
from hemo1d.solvers.cg import CGFEMDiscretization, CGMeshConfig, CGScalarMassSolver


@pytest.fixture
def physics() -> Hemo1DPhysics:
    params = ModelParameters(
        blood=BloodParameters(rho=1.06, mu=0.035),
        vessel=VesselParameters(
            length=2.0,
            area0=0.126,
            beta=0.060606e7,
        ),
        gamma_profile=2.0,
        p0=0.0,
        p_ext=0.0,
    )
    return Hemo1DPhysics(params, NP_BACKEND)


def test_mass_solver_recovers_constant_with_matching_endpoint_values(
    physics: Hemo1DPhysics,
):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Mass solver endpoint test is serial-only for now.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    u_exact = fem.Function(disc.V)
    u_exact.interpolate(lambda x: np.full(x.shape[1], 3.5, dtype=np.float64))
    u_exact.x.scatter_forward()

    v = ufl.TestFunction(disc.V)
    dx = ufl.dx(domain=disc.domain)
    rhs_form = fem.form(u_exact * v * dx)

    out = fem.Function(disc.V)

    solver = CGScalarMassSolver(disc)
    solver.solve(
        rhs_form=rhs_form,
        out=out,
        left_value=3.5,
        right_value=3.5,
    )

    assert np.allclose(out.x.array, 3.5, rtol=1.0e-12, atol=1.0e-12)


def test_mass_solver_imposes_endpoint_values(
    physics: Hemo1DPhysics,
):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Mass solver endpoint test is serial-only for now.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )

    zero = fem.Function(disc.V)
    zero.interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
    zero.x.scatter_forward()

    v = ufl.TestFunction(disc.V)
    dx = ufl.dx(domain=disc.domain)
    rhs_form = fem.form(zero * v * dx)

    out = fem.Function(disc.V)

    solver = CGScalarMassSolver(disc)
    solver.solve(
        rhs_form=rhs_form,
        out=out,
        left_value=1.25,
        right_value=-0.75,
    )

    left_dof = int(disc.endpoint_dofs(EndpointSide.LEFT)[0])
    right_dof = int(disc.endpoint_dofs(EndpointSide.RIGHT)[0])

    assert math.isclose(out.x.array[left_dof], 1.25, rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(out.x.array[right_dof], -0.75, rel_tol=1e-14, abs_tol=1e-14)


def test_mass_solver_recovers_smooth_function_when_endpoints_match(
    physics: Hemo1DPhysics,
):
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("Mass solver endpoint test is serial-only for now.")

    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=32, degree=1)
    )

    L = physics.params.length

    u_exact = fem.Function(disc.V)
    u_exact.interpolate(lambda x: 2.0 + np.sin(np.pi * x[0] / L))
    u_exact.x.scatter_forward()

    v = ufl.TestFunction(disc.V)
    dx = ufl.dx(domain=disc.domain)
    rhs_form = fem.form(u_exact * v * dx)

    out = fem.Function(disc.V)

    solver = CGScalarMassSolver(disc)
    solver.solve(
        rhs_form=rhs_form,
        out=out,
        left_value=2.0,
        right_value=2.0,
    )

    assert np.allclose(out.x.array, u_exact.x.array, rtol=1.0e-11, atol=1.0e-11)


def test_mass_solver_rejects_mpi_endpoint_ambiguity_only_if_run_parallel(
    physics: Hemo1DPhysics,
):
    """
    This is mostly a placeholder documenting the current limitation.

    The solver is intentionally serial-oriented for endpoint Dirichlet handling.
    Later we will make endpoint ownership MPI-safe.
    """
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )

    if MPI.COMM_WORLD.size == 1:
        solver = CGScalarMassSolver(disc)
        assert solver is not None
    else:
        # Under MPI, depending on mesh ownership, some ranks may not own endpoint dofs.
        # We do not enforce behavior yet.
        pass