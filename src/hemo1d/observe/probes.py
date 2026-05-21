from __future__ import annotations

from typing import Any

from hemo1d.observe.history import ProbePoint, ProbeSample


class NetworkProbeRecorder:
    """
    Probe recorder for a multi-vessel network.

    This recorder is discretization-agnostic. It only assumes each vessel has:

        vessel.sample_state(coordinate)
        vessel.physics.pressure(area)

    Therefore it works for CG now and should work for DG later, as long as DG
    vessels implement the same Vessel interface.
    """

    def __init__(
        self,
        *,
        vessels: dict[str, Any],
        probes: list[ProbePoint],
    ) -> None:
        self.vessels = vessels
        self.probes = probes

        self._validate_probes()

    def _validate_probes(self) -> None:
        for probe in self.probes:
            if probe.vessel_id not in self.vessels:
                raise ValueError(
                    f"Probe {probe.name!r} refers to unknown vessel "
                    f"{probe.vessel_id!r}."
                )

            vessel = self.vessels[probe.vessel_id]

            if probe.coordinate < 0.0 or probe.coordinate > vessel.length:
                raise ValueError(
                    f"Probe {probe.name!r} on vessel {probe.vessel_id!r} has coordinate "
                    f"{probe.coordinate}, outside [0, {vessel.length}]."
                )

    def sample(self, time: float) -> list[ProbeSample]:
        samples: list[ProbeSample] = []

        for probe in self.probes:
            vessel = self.vessels[probe.vessel_id]
            point_state = vessel.sample_state(probe.coordinate)

            pressure = float(vessel.physics.pressure(point_state.area))

            samples.append(
                ProbeSample(
                    time=time,
                    vessel_id=probe.vessel_id,
                    name=probe.name,
                    coordinate=probe.coordinate,
                    area=point_state.area,
                    flow_rate=point_state.flow_rate,
                    pressure=pressure,
                )
            )

        return samples