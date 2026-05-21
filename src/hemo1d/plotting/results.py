from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_current_figure(
    output_dir: Path,
    filename: str,
    dpi: int = 200,
    *,
    close: bool = False,
) -> None:
    """Save the current matplotlib figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=dpi)
    if close:
        plt.close()


def plot_vessel_probe_histories(
    *,
    output_dir: Path,
    result: Any,
    vessel_id: str,
    probe_names: list[str],
    show_area: bool = True,
    show_flow_rate: bool = True,
    show_pressure: bool = True,
    close: bool = False,
) -> None:
    """Plot recorded probe histories for one vessel."""

    if show_area:
        _plot_probe_quantity_history(
            output_dir=output_dir,
            result=result,
            vessel_id=vessel_id,
            probe_names=probe_names,
            quantity_name="area",
            y_label="A [cm^2]",
            title=f"Area histories: {vessel_id}",
            filename=f"{vessel_id}_area_history.png",
            close=close,
        )

    if show_flow_rate:
        _plot_probe_quantity_history(
            output_dir=output_dir,
            result=result,
            vessel_id=vessel_id,
            probe_names=probe_names,
            quantity_name="flow_rate",
            y_label="Q [cm^3/s]",
            title=f"Flow-rate histories: {vessel_id}",
            filename=f"{vessel_id}_flow_rate_history.png",
            close=close,
        )

    if show_pressure:
        _plot_probe_quantity_history(
            output_dir=output_dir,
            result=result,
            vessel_id=vessel_id,
            probe_names=probe_names,
            quantity_name="pressure",
            y_label="P",
            title=f"Pressure histories: {vessel_id}",
            filename=f"{vessel_id}_pressure_history.png",
            close=close,
        )


def plot_junction_flow_split(
    *,
    output_dir: Path,
    result: Any,
    parent_vessel_id: str = "parent",
    daughter1_vessel_id: str = "daughter1",
    daughter2_vessel_id: str = "daughter2",
    junction_probe_name: str = "junction",
    close: bool = False,
) -> None:
    """Plot the standard three-vessel junction flow split from probes."""

    parent_samples = result.history.probes.by_vessel_and_name(
        parent_vessel_id,
        junction_probe_name,
    )
    d1_samples = result.history.probes.by_vessel_and_name(
        daughter1_vessel_id,
        junction_probe_name,
    )
    d2_samples = result.history.probes.by_vessel_and_name(
        daughter2_vessel_id,
        junction_probe_name,
    )

    if not (parent_samples and d1_samples and d2_samples):
        return

    times = np.array([sample.time for sample in parent_samples])
    q_parent = np.array([sample.flow_rate for sample in parent_samples])
    q_d1 = np.array([sample.flow_rate for sample in d1_samples])
    q_d2 = np.array([sample.flow_rate for sample in d2_samples])

    plt.figure()
    plt.plot(times, q_parent, label=f"Q {parent_vessel_id} at junction")
    plt.plot(times, q_d1, label=f"Q {daughter1_vessel_id} at junction")
    plt.plot(times, q_d2, label=f"Q {daughter2_vessel_id} at junction")
    plt.plot(times, q_d1 + q_d2, "--", label="Q daughter1 + Q daughter2")
    plt.xlabel("t [s]")
    plt.ylabel("Q [cm^3/s]")
    plt.title("Junction flow split")
    plt.grid(True)
    plt.legend()
    save_current_figure(output_dir, "junction_flow_split.png", close=close)

    plt.figure()
    plt.plot(times, q_parent - q_d1 - q_d2)
    plt.xlabel("t [s]")
    plt.ylabel("Q parent - Q daughter1 - Q daughter2")
    plt.title("Junction mass residual from probes")
    plt.grid(True)
    save_current_figure(
        output_dir,
        "junction_mass_residual_from_probes.png",
        close=close,
    )


def _plot_probe_quantity_history(
    *,
    output_dir: Path,
    result: Any,
    vessel_id: str,
    probe_names: list[str],
    quantity_name: str,
    y_label: str,
    title: str,
    filename: str,
    close: bool,
) -> None:
    plotted_any = False

    plt.figure()
    for probe_name in probe_names:
        samples = result.history.probes.by_vessel_and_name(vessel_id, probe_name)
        if not samples:
            continue

        plotted_any = True
        times = np.array([sample.time for sample in samples])
        values = np.array([getattr(sample, quantity_name) for sample in samples])
        plt.plot(times, values, label=probe_name)

    if plotted_any:
        plt.xlabel("t [s]")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        save_current_figure(output_dir, filename, close=close)
    else:
        plt.close()
