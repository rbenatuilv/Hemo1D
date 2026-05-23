from __future__ import annotations

from collections.abc import Callable

import numpy as np


def create_sinusoidal_inflow(
    amplitude: float,
    period: float,
    ramp_time: float = 0.0,
) -> Callable[[float], float]:
    """Create a non-negative sinusoidal flow-rate function."""

    if period <= 0.0:
        raise ValueError("period must be positive.")
    if ramp_time < 0.0:
        raise ValueError("ramp_time must be non-negative.")

    def q(t: float) -> float:
        value = amplitude * (1.0 - np.cos(2.0 * np.pi * t / period)) / 2.0
        if ramp_time > 0.0 and t < ramp_time:
            return (t / ramp_time) * value
        return value

    return q


def create_positive_sine_inflow(
    amplitude: float,
    duration: float,
) -> Callable[[float], float]:
    """Create a compact positive half-sine flow-rate pulse."""

    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    def q(t: float) -> float:
        if 0.0 <= t <= duration:
            return float(amplitude * np.sin(np.pi * t / duration))
        return 0.0

    return q


def create_periodic_positive_sine_inflow(
    amplitude: float,
    duration: float,
    period: float,
) -> Callable[[float], float]:
    """Create a periodic positive half-sine flow-rate function."""

    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    def q(t: float) -> float:
        t_mod = t % period
        if 0.0 <= t_mod <= duration:
            return float(amplitude * np.sin(np.pi * t_mod / duration))
        else:
            return 0.0

    return q


def create_pulsatile_inflow(
    systolic_amplitude: float,
    systolic_duration: float,
    cycle_period: float,
) -> Callable[[float], float]:
    """Create a simple periodic cardiac-like flow-rate waveform."""

    if systolic_duration <= 0.0:
        raise ValueError("systolic_duration must be positive.")
    if cycle_period <= systolic_duration:
        raise ValueError("cycle_period must be larger than systolic_duration.")

    def q(t: float) -> float:
        t_mod = t % cycle_period
        if t_mod < systolic_duration:
            return float(
                systolic_amplitude * np.sin(np.pi * t_mod / systolic_duration)
            )

        diastolic_time = t_mod - systolic_duration
        diastolic_duration = cycle_period - systolic_duration
        decay_factor = np.exp(-3.0 * diastolic_time / diastolic_duration)
        return float(systolic_amplitude * 0.2 * decay_factor)

    return q


__all__ = [
    "create_positive_sine_inflow",
    "create_pulsatile_inflow",
    "create_sinusoidal_inflow",
    "create_periodic_positive_sine_inflow",
]
