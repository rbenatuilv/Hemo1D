from __future__ import annotations

from dataclasses import dataclass

from hemo1d.core.state import EndpointSide


@dataclass(frozen=True)
class NetworkEndpoint:
    """
    Identifier for one endpoint of one vessel in a network.

    This is used as the key for:
        - external boundary conditions,
        - junction endpoint states,
        - endpoint-state maps during time stepping.
    """

    vessel_id: str
    side: EndpointSide

    @property
    def is_left(self) -> bool:
        return self.side == EndpointSide.LEFT

    @property
    def is_right(self) -> bool:
        return self.side == EndpointSide.RIGHT

    @property
    def outward_normal_sign(self) -> int:
        """
        Sign of the outward normal in the local vessel coordinate.

        LEFT endpoint:
            outward normal = -1

        RIGHT endpoint:
            outward normal = +1
        """
        if self.side == EndpointSide.LEFT:
            return -1

        return 1

    def outward_flow(self, flow_rate: float) -> float:
        """
        Flow rate leaving the vessel through this endpoint.

        If Q is positive in the increasing-z direction:

            LEFT endpoint:
                outward flow = -Q

            RIGHT endpoint:
                outward flow = +Q
        """
        return self.outward_normal_sign * flow_rate

    def label(self) -> str:
        return f"{self.vessel_id}.{self.side.value}"