import numpy as np
import pytest

from hemo1d.solvers.model_solver import NetworkSolver
from hemo1d.solvers.time import TimeConfig


class FakePhysics:
    def pressure(self, area):
        return np.zeros_like(area, dtype=float)

    def wave_speed(self, area):
        return np.ones_like(area, dtype=float)


class FakeVessel:
    vessel_id = "vessel"
    physics = FakePhysics()

    def state_arrays(self):
        return (
            np.array([0.0, 1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )


class FakeNetwork:
    vessels = {"vessel": FakeVessel()}
    lumped_beds = []

    def require_complete(self):
        return None


class RecordingNetworkSolver(NetworkSolver):
    def __init__(self):
        super().__init__(FakeNetwork())
        self.step_sizes = []

    def step(self, *, time: float, dt: float) -> None:
        self.step_sizes.append(dt)


def test_network_solver_skips_tiny_final_remainder_step():
    solver = RecordingNetworkSolver()
    t_end = 2.0e-5 + 5.0e-13

    result = solver.run(
        config=TimeConfig(
            t0=0.0,
            t_end=t_end,
            fixed_dt=1.0e-5,
        ),
        record_every=1,
        show_progress=False,
    )

    assert result.time == t_end
    assert result.num_steps == 2
    assert solver.step_sizes == pytest.approx([1.0e-5, 1.0e-5])
    assert result.history.times[-1] == t_end
