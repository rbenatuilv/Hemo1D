from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProbePoint:
    """
    A named sampling point inside one vessel.

    vessel_id:
        Identifier of the vessel where the probe is located.

    name:
        Human-readable name, e.g. "inlet", "junction", "outlet".

    coordinate:
        Local vessel coordinate z.
    """

    vessel_id: str
    name: str
    coordinate: float


@dataclass(frozen=True)
class ProbeSample:
    """
    One time sample at one probe point.
    """

    time: float
    vessel_id: str
    name: str
    coordinate: float
    area: float
    flow_rate: float
    pressure: float


@dataclass
class ProbeHistory:
    """
    Collection of probe samples over time.
    """

    samples: list[ProbeSample] = field(default_factory=list)

    def by_name(self, name: str) -> list[ProbeSample]:
        return [sample for sample in self.samples if sample.name == name]

    def by_vessel(self, vessel_id: str) -> list[ProbeSample]:
        return [sample for sample in self.samples if sample.vessel_id == vessel_id]

    def by_vessel_and_name(self, vessel_id: str, name: str) -> list[ProbeSample]:
        return [
            sample
            for sample in self.samples
            if sample.vessel_id == vessel_id and sample.name == name
        ]

    def names(self) -> list[str]:
        return sorted(set(sample.name for sample in self.samples))

    def vessel_ids(self) -> list[str]:
        return sorted(set(sample.vessel_id for sample in self.samples))

    def keys(self) -> list[tuple[str, str]]:
        """
        Return sorted (vessel_id, probe_name) pairs.
        """
        return sorted(set((sample.vessel_id, sample.name) for sample in self.samples))