import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc

from hemo1d.core.backend import NP_BACKEND
from hemo1d.core.parameters import BloodParameters, ModelParameters, VesselParameters
from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.cg import (
    CGFEMDiscretization,
    CGMeshConfig,
    CGTaylorGalerkinFormBuilder,
)


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


def assemble_vector_array(form: fem.Form) -> np.ndarray:
    vec = fem_petsc.assemble_vector(form)
    vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    return vec.array.copy()


def test_form_builder_rejects_non_positive_dt(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )
    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    builder = CGTaylorGalerkinFormBuilder(disc, physics)

    with pytest.raises(ValueError):
        builder.build(state_n=state, dt=0.0)

    with pytest.raises(ValueError):
        builder.build(state_n=state, dt=-1.0e-5)


def test_forms_assemble_at_rest_state(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=8, degree=1)
    )
    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    builder = CGTaylorGalerkinFormBuilder(disc, physics)
    forms = builder.build(state_n=state, dt=1.0e-5)

    rhs_A = assemble_vector_array(forms.rhs_A)
    rhs_Q = assemble_vector_array(forms.rhs_Q)

    assert np.all(np.isfinite(rhs_A))
    assert np.all(np.isfinite(rhs_Q))


def test_rhs_preserves_rest_state_before_boundary_terms(physics: Hemo1DPhysics):
    """
    At rest:

        A = A0
        Q = 0

    and for constant A0, beta:

        S = 0
        dF/dz = 0

    Therefore the Taylor-Galerkin RHS should reduce to:

        rhs_A = (A0, v)
        rhs_Q = (0, v)

    This is a key consistency check.
    """
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )
    state = disc.create_state()
    disc.interpolate_rest_state(state, physics)

    builder = CGTaylorGalerkinFormBuilder(disc, physics)
    forms = builder.build(state_n=state, dt=1.0e-5)

    rhs_A = assemble_vector_array(forms.rhs_A)
    rhs_Q = assemble_vector_array(forms.rhs_Q)

    v = ufl.TestFunction(disc.V)
    dx = ufl.dx(domain=disc.domain)

    reference_A_form = fem.form(state.A * v * dx)
    reference_Q_form = fem.form(state.Q * v * dx)

    reference_A = assemble_vector_array(reference_A_form)
    reference_Q = assemble_vector_array(reference_Q_form)

    assert np.allclose(rhs_A, reference_A, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(rhs_Q, reference_Q, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(rhs_Q, 0.0, rtol=1.0e-12, atol=1.0e-12)


def test_forms_assemble_for_smooth_area_perturbation(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )
    state = disc.create_state()

    A0 = physics.params.area0
    L = physics.params.length

    state.A.interpolate(
        lambda x: A0 * (1.0 + 1.0e-3 * np.sin(np.pi * x[0] / L))
    )
    state.Q.interpolate(lambda x: np.zeros(x.shape[1], dtype=np.float64))
    state.scatter_forward()

    builder = CGTaylorGalerkinFormBuilder(disc, physics)
    forms = builder.build(state_n=state, dt=1.0e-5)

    rhs_A = assemble_vector_array(forms.rhs_A)
    rhs_Q = assemble_vector_array(forms.rhs_Q)

    assert np.all(np.isfinite(rhs_A))
    assert np.all(np.isfinite(rhs_Q))


def test_forms_assemble_for_smooth_flow_perturbation(physics: Hemo1DPhysics):
    disc = CGFEMDiscretization(
        CGMeshConfig(length=physics.params.length, num_cells=16, degree=1)
    )
    state = disc.create_state()

    A0 = physics.params.area0
    L = physics.params.length

    state.A.interpolate(lambda x: np.full(x.shape[1], A0, dtype=np.float64))
    state.Q.interpolate(
        lambda x: 1.0e-3 * np.sin(np.pi * x[0] / L)
    )
    state.scatter_forward()

    builder = CGTaylorGalerkinFormBuilder(disc, physics)
    forms = builder.build(state_n=state, dt=1.0e-5)

    rhs_A = assemble_vector_array(forms.rhs_A)
    rhs_Q = assemble_vector_array(forms.rhs_Q)

    assert np.all(np.isfinite(rhs_A))
    assert np.all(np.isfinite(rhs_Q))