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


def create_arterial_pressure_inflow(
    mean_mmhg: float,
    amp1_mmhg: float,
    amp2_mmhg: float,
    heart_rate: float = 1.2,
    ramp_time: float = 1.0,
    mmhg_to_dyn_cm2: float = 1333.22,
    phase_s: float = 0.0,
) -> Callable[[float], float]:
    """Create a physiological arterial pressure waveform.
    
    Parameters
    ----------
    mean_mmhg : float
        Mean arterial pressure in mmHg.
    amp1_mmhg : float
        Amplitude of the fundamental frequency component in mmHg.
    amp2_mmhg : float
        Amplitude of the second harmonic component in mmHg.
    heart_rate : float, optional
        Heart rate in Hz (beats per second). Default is 1.2 Hz (72 bpm).
    ramp_time : float, optional
        Ramp-up time in seconds to smooth startup transient. Default is 1.0 s.
    mmhg_to_dyn_cm2 : float, optional
        Conversion factor from mmHg to dyn/cm². Default is 1333.22.
    phase_s : float, optional
        Phase shift in seconds. Default is 0.0.
    """
    if heart_rate <= 0.0:
        raise ValueError("heart_rate must be positive.")
    if ramp_time < 0.0:
        raise ValueError("ramp_time must be non-negative.")

    def p(t: float) -> float:
        return arterial_pressure(
            t,
            mean_mmhg=mean_mmhg,
            amp1_mmhg=amp1_mmhg,
            amp2_mmhg=amp2_mmhg,
            heart_rate=heart_rate,
            ramp_time=ramp_time,
            mmhg_to_dyn_cm2=mmhg_to_dyn_cm2,
            phase_s=phase_s,
        )

    return p


def mmhg(value_dyn_cm2: float, mmhg_to_dyn_cm2: float = 1333.22) -> float:
    return value_dyn_cm2 * mmhg_to_dyn_cm2


def ramp(t: float, ramp_time: float = 1.0) -> float:
    """Smooth enough startup for the pulsatile part."""
    if ramp_time <= 0.0:
        return 1.0
    return min(1.0, max(0.0, t / ramp_time))


def arterial_pressure(
    t: float,
    *,
    mean_mmhg: float,
    amp1_mmhg: float,
    amp2_mmhg: float,
    heart_rate: float = 1.2,
    ramp_time: float = 1.0,
    mmhg_to_dyn_cm2: float = 1333.22,
    phase_s: float = 0.0,
) -> float:
    """
    Simple physiological pressure waveform.

    Mean pressure is always present.
    Pulsatility is gradually ramped in to avoid a violent startup transient.
    """
    tau = t - phase_s
    omega = 2.0 * np.pi * heart_rate

    pulsatile_mmhg = (
        amp1_mmhg * np.sin(omega * tau)
        + amp2_mmhg * np.sin(2.0 * omega * tau - 0.8)
    )

    pressure_mmhg = mean_mmhg + ramp(t, ramp_time) * pulsatile_mmhg
    return mmhg(pressure_mmhg, mmhg_to_dyn_cm2)



__all__ = [
    "create_positive_sine_inflow",
    "create_pulsatile_inflow",
    "create_sinusoidal_inflow",
    "create_periodic_positive_sine_inflow",
    "create_arterial_pressure_inflow",
]
