from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeConfig:
    """
    Time integration configuration.

    If fixed_dt is None, the solver computes dt from the CFL condition at each step.
    """

    t0: float
    t_end: float
    fixed_dt: float | None = None
    cfl: float = np.sqrt(3.0) / 3.0
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        if self.t_end <= self.t0:
            raise ValueError("t_end must be larger than t0.")

        if self.fixed_dt is not None and self.fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be positive if provided.")

        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive.")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")