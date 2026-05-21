from __future__ import annotations

from typing import Any

import numpy as np

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.core.state import StateAtPoint
from hemo1d.solvers.cg.discretization import CGFEMDiscretization
from hemo1d.solvers.cg.stepper import CGTaylorGalerkinStepper
from hemo1d.solvers.vessel import Vessel


def sample_cg_state_linear(
    discretization: CGFEMDiscretization,
    state: Any,
    physics: Hemo1DPhysics,
    coordinate: float,
) -> StateAtPoint:
    """
    Sample a CG state using interpolation over nodal values.

    This function is CG-specific but uses only the public discretization API.
    The underlying DOF coordinates are cached in the discretization for efficiency.
    """
    # Obtain sorted DOF indices and coordinates once (cached by discretization)
    dofs, z_sorted = discretization.dof_coordinates_sorted()

    area_sorted = state.A.x.array[dofs]
    flow_sorted = state.Q.x.array[dofs]

    area = float(np.interp(coordinate, z_sorted, area_sorted))
    flow_rate = float(np.interp(coordinate, z_sorted, flow_sorted))

    return StateAtPoint(
        area=area,
        flow_rate=flow_rate,
    )


def extract_cg_state_arrays(
    discretization: CGFEMDiscretization,
    state: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract nodal arrays from a CG state.

    This function is CG-specific but uses only the public discretization API.
    The underlying DOF coordinates are cached in the discretization for efficiency.

    Returns:
        z, A, Q

    The arrays are sorted by increasing z.
    """
    dofs, z = discretization.dof_coordinates_sorted()

    return (
        z.copy(),
        state.A.x.array[dofs].copy(),
        state.Q.x.array[dofs].copy(),
    )


def create_cg_vessel(
    *,
    vessel_id: str,
    physics: Hemo1DPhysics,
    discretization: CGFEMDiscretization,
) -> Vessel:
    """
    Create a generic Vessel using the current CG Taylor-Galerkin implementation.

    The returned object is generic, but the stepper, sampler, and array
    extractor are CG-specific.
    """
    stepper = CGTaylorGalerkinStepper(discretization, physics)

    state_n = discretization.create_state(name=f"{vessel_id}_n")
    state_np1 = discretization.create_state(name=f"{vessel_id}_np1")

    return Vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=discretization,
        stepper=stepper,
        state_n=state_n,
        state_np1=state_np1,
        state_sampler=sample_cg_state_linear,
        state_array_extractor=extract_cg_state_arrays,
    )
