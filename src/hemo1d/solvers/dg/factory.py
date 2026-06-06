from __future__ import annotations

from hemo1d.core.physics import Hemo1DPhysics
from hemo1d.solvers.dg.discretization import DGFEMDiscretization
from hemo1d.solvers.dg.flux import DGFluxScheme
from hemo1d.solvers.dg.limiter import DGLimiterConfig
from hemo1d.solvers.dg.sampling import (
    extract_dg_state_arrays,
    sample_dg_state_array,
    sample_dg_state,
)
from hemo1d.solvers.dg.stepper import DGLaxFriedrichsStepper
from hemo1d.solvers.vessel import Vessel


def create_dg_vessel(
    *,
    vessel_id: str,
    physics: Hemo1DPhysics,
    discretization: DGFEMDiscretization,
    time_scheme: str = "rk2",
    limiter_config: DGLimiterConfig | None = None,
    flux_scheme: DGFluxScheme | str = "lxf",
) -> Vessel:
    """
    Create a generic Vessel using the array-based DG implementation.

    By default:
        - RK2 time stepping
        - slope limiter enabled
        - positivity limiter enabled
        - local Lax-Friedrichs/Rusanov interface flux

    For smooth convergence tests, you can pass a less aggressive limiter config,
    for example:

        DGLimiterConfig(
            enabled=True,
            slope=True,
            positivity=True,
            minmod_beta=2.0,
        )

    or disable it entirely for pure smooth tests:

        DGLimiterConfig(enabled=False)
    """
    stepper = DGLaxFriedrichsStepper(
        discretization=discretization,
        physics=physics,
        time_scheme=time_scheme,  # type: ignore[arg-type]
        limiter_config=limiter_config,
        flux_scheme=flux_scheme,
    )

    state_n = discretization.create_state(name=f"{vessel_id}_n")
    state_np1 = discretization.create_state(name=f"{vessel_id}_np1")

    return Vessel(
        vessel_id=vessel_id,
        physics=physics,
        discretization=discretization,
        stepper=stepper,
        state_n=state_n,
        state_np1=state_np1,
        state_sampler=sample_dg_state,
        state_array_extractor=extract_dg_state_arrays,
        state_array_sampler=sample_dg_state_array,
    )
