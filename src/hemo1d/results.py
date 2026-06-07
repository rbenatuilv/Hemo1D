from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hemo1d.io import (
    write_diagnostics_csv,
    write_probe_history_csv,
    write_vessel_final_state_csv,
)


@dataclass
class Results:
    """High-level simulation result returned by ``HemodynamicModel.solve``."""

    raw: Any
    solver_settings: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def network(self) -> Any:
        return self.raw.network

    @property
    def history(self) -> Any:
        return self.raw.history

    @property
    def time(self) -> float:
        return float(self.raw.time)

    @property
    def num_steps(self) -> int:
        return int(self.raw.num_steps)

    def capillary_bed_history(self, bed_id: str) -> list[Any]:
        """Return diagnostic samples for one lumped capillary bed."""

        samples = [
            sample.lumped_bed_samples[bed_id]
            for sample in self.history.diagnostics
            if bed_id in getattr(sample, "lumped_bed_samples", {})
        ]
        if not samples:
            known = sorted(
                {
                    bed
                    for sample in self.history.diagnostics
                    for bed in getattr(sample, "lumped_bed_samples", {})
                }
            )
            raise KeyError(
                f"No capillary bed history for {bed_id!r}. "
                f"Available beds: {known}."
            )
        return samples

    def capillary_bed_ids(self) -> list[str]:
        """Return ids for recorded lumped capillary beds."""

        return sorted(
            {
                bed_id
                for sample in self.history.diagnostics
                for bed_id in getattr(sample, "lumped_bed_samples", {})
            }
        )

    def capillary_bed_pressure(self, bed_id: str) -> np.ndarray:
        """Return the recorded bed pressure time series."""

        return np.array(
            [sample.pressure for sample in self.capillary_bed_history(bed_id)],
            dtype=float,
        )

    def regional_perfusion(self, bed_id: str) -> np.ndarray:
        """Return regional perfusion history for a bed with tissue_volume set."""

        samples = self.capillary_bed_history(bed_id)
        if any(sample.regional_perfusion is None for sample in samples):
            raise ValueError(
                f"Capillary bed {bed_id!r} has no regional perfusion history; "
                "set tissue_volume when creating the bed."
            )
        return np.array([sample.regional_perfusion for sample in samples], dtype=float)

    def save_probes(self, path: str | Path) -> None:
        """Save probe time histories to CSV."""

        write_probe_history_csv(self.history, path)

    def save(self, path: str | Path) -> None:
        """Save probes, diagnostics, final vessel states, and metadata."""

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.save_probes(output_dir / "probes.csv")
        write_diagnostics_csv(self.history, output_dir / "diagnostics.csv")

        final_state_dir = output_dir / "final_states"
        final_state_dir.mkdir(parents=True, exist_ok=True)
        for vessel in self.network.vessels.values():
            write_vessel_final_state_csv(
                vessel,
                final_state_dir / f"{vessel.vessel_id}_final_state.csv",
            )

        metadata = {
            **self.metadata,
            "time": self.time,
            "num_steps": self.num_steps,
            "solver": _jsonable(self.solver_settings),
            "vessels": sorted(self.network.vessels),
        }
        with (output_dir / "metadata.json").open("w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

    def plot_probes(
        self,
        output_dir: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot all recorded probe histories, optionally saving figures."""

        if output_dir is None:
            self._plot_probes_interactive(show=show)
            return

        from hemo1d.plotting import plot_vessel_probe_histories

        path = Path(output_dir)
        for vessel_id in self.history.probes.vessel_ids():
            probe_names = [
                name
                for probe_vessel, name in self.history.probes.keys()
                if probe_vessel == vessel_id
            ]
            plot_vessel_probe_histories(
                output_dir=path,
                result=self.raw,
                vessel_id=vessel_id,
                probe_names=probe_names,
                close=not show,
            )

        if show:
            import matplotlib.pyplot as plt

            plt.show()

    def plot_capillary_beds(
        self,
        output_dir: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot all recorded lumped capillary-bed histories."""

        bed_ids = self.capillary_bed_ids()
        if not bed_ids:
            return

        from hemo1d.plotting import plot_capillary_bed_history

        path = None if output_dir is None else Path(output_dir)
        for bed_id in bed_ids:
            plot_capillary_bed_history(
                output_dir=path,
                samples=self.capillary_bed_history(bed_id),
                bed_id=bed_id,
                close=not show,
            )

        if show:
            import matplotlib.pyplot as plt

            plt.show()

    def _plot_probes_interactive(self, *, show: bool) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        for vessel_id in self.history.probes.vessel_ids():
            probe_names = [
                name
                for probe_vessel, name in self.history.probes.keys()
                if probe_vessel == vessel_id
            ]
            for quantity, y_label in (
                ("area", "A [cm^2]"),
                ("flow_rate", "Q [cm^3/s]"),
                ("pressure", "P"),
            ):
                plotted_any = False
                plt.figure()
                for probe_name in probe_names:
                    samples = self.history.probes.by_vessel_and_name(
                        vessel_id,
                        probe_name,
                    )
                    if not samples:
                        continue
                    plotted_any = True
                    times = np.array([sample.time for sample in samples])
                    values = np.array([getattr(sample, quantity) for sample in samples])
                    plt.plot(times, values, label=probe_name)

                if plotted_any:
                    plt.xlabel("t [s]")
                    plt.ylabel(y_label)
                    plt.title(f"{quantity.replace('_', ' ').title()}: {vessel_id}")
                    plt.grid(True)
                    plt.legend()
                else:
                    plt.close()

        if show:
            plt.show()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
