from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hemo1d.convergence.network_snapshots import (
    NetworkSnapshotHistory,
    NetworkSnapshotRecorder,
)
from hemo1d.core.state import BoundaryState
from hemo1d.observe.diagnostics import StateDiagnostics, compute_vessel_diagnostics
from hemo1d.topology.endpoint import NetworkEndpoint
from hemo1d.topology.graph import Junction, VascularNetwork
from hemo1d.boundary.junction import (
    JunctionData,
    JunctionEndpointData,
    JunctionSolver,
)
from hemo1d.solvers.vessel import Vessel
from hemo1d.observe import NetworkProbeRecorder, ProbeHistory, ProbePoint
from hemo1d.solvers.time import TimeConfig


@dataclass
class NetworkDiagnosticsSample:
    """
    Diagnostics of all vessels at a given time.
    """

    time: float
    vessel_diagnostics: dict[str, StateDiagnostics]


@dataclass
class NetworkHistory:
    """
    Network time history.

    Stores:
        - global diagnostics per vessel,
        - optional probe samples.
        - optional spatial snapshots for convergence studies.
    """

    diagnostics: list[NetworkDiagnosticsSample] = field(default_factory=list)
    probes: ProbeHistory = field(default_factory=ProbeHistory)
    snapshots: NetworkSnapshotHistory = field(default_factory=NetworkSnapshotHistory)

    @property
    def times(self) -> list[float]:
        return [sample.time for sample in self.diagnostics]


@dataclass
class NetworkSolverResult:
    """
    Result returned by NetworkSolver.run().
    """

    network: VascularNetwork
    time: float
    num_steps: int
    history: NetworkHistory


class NetworkSolver:
    """
    Generic network solver.

    This solver handles:
        - a single vessel,
        - one junction,
        - multiple junctions,
        - larger networks.

    Each vessel may later use a different discretization, as long as it exposes
    the Vessel interface.
    """

    def __init__(
        self,
        network: VascularNetwork,
        junction_solver: JunctionSolver | None = None,
    ) -> None:
        network.require_complete()

        self.network = network
        self.junction_solver = junction_solver if junction_solver is not None else JunctionSolver()

    def vessels(self) -> dict[str, Vessel]:
        return self.network.vessels

    def compute_dt(self, time: float, config: TimeConfig) -> float:
        """
        Compute global time step.

        If config.fixed_dt is provided, use it. Otherwise compute the minimum
        stable time step across all vessels.
        """
        if config.fixed_dt is not None:
            dt = config.fixed_dt
        else:
            dt = min(
                vessel.compute_stable_dt(config.cfl) for vessel in self.network.vessels.values()
            )

        return min(dt, config.t_end - time)

    def record_diagnostics(self, history: NetworkHistory, time: float) -> None:
        history.diagnostics.append(
            NetworkDiagnosticsSample(
                time=time,
                vessel_diagnostics={
                    vessel_id: compute_vessel_diagnostics(vessel, time)
                    for vessel_id, vessel in self.network.vessels.items()
                },
            )
        )

    def compute_external_boundary_states(
        self,
        *,
        t_np1: float,
        dt: float,
    ) -> dict[NetworkEndpoint, BoundaryState]:
        """
        Compute BoundaryState values for all external boundaries.
        """
        endpoint_states: dict[NetworkEndpoint, BoundaryState] = {}

        for endpoint, boundary in self.network.external_boundaries.items():
            vessel = self.network.vessels[endpoint.vessel_id]
            endpoint_data = vessel.endpoint_data(endpoint.side)

            endpoint_states[endpoint] = boundary.compute(
                physics=vessel.physics,
                endpoint_data=endpoint_data,
                side=endpoint.side,
                t=t_np1,
                dt=dt,
            )

        return endpoint_states

    def solve_junction(
        self,
        junction: Junction,
        dt: float,
    ) -> dict[NetworkEndpoint, BoundaryState]:
        """
        Solve one two- or three-vessel junction and return endpoint states.
        """
        data = JunctionData(
            endpoints=tuple(
                self._junction_endpoint_data(endpoint, angle)
                for endpoint, angle in zip(junction.endpoints, junction.angles, strict=True)
            )
        )

        solution = self.junction_solver.solve(
            data=data,
            dt=dt,
            raise_on_failure=True,
        )

        return dict(zip(junction.endpoints, solution.endpoint_states, strict=True))

    def _junction_endpoint_data(
        self,
        endpoint: NetworkEndpoint,
        angle: float | None,
    ) -> JunctionEndpointData:
        vessel = self.network.vessels[endpoint.vessel_id]
        return JunctionEndpointData(
            physics=vessel.physics,
            endpoint_data=vessel.endpoint_data(endpoint.side),
            side=endpoint.side,
            name=endpoint.label(),
            angle=angle,
        )

    def solve_all_junctions(
        self,
        dt: float,
    ) -> dict[NetworkEndpoint, BoundaryState]:
        """
        Solve all junction systems independently.

        This assumes junctions are coupled only through vessel states at the
        previous time step, which matches the current explicit/semi-explicit
        time stepping.
        """
        endpoint_states: dict[NetworkEndpoint, BoundaryState] = {}

        for junction in self.network.junctions:
            solved = self.solve_junction(
                junction=junction,
                dt=dt,
            )

            overlap = set(endpoint_states) & set(solved)
            if overlap:
                labels = sorted(endpoint.label() for endpoint in overlap)
                raise RuntimeError(f"Duplicate junction endpoint states computed for: {labels}.")

            endpoint_states.update(solved)

        return endpoint_states

    def compute_endpoint_states(
        self,
        *,
        t_np1: float,
        dt: float,
    ) -> dict[NetworkEndpoint, BoundaryState]:
        """
        Compute BoundaryState for every endpoint in the network.
        """
        endpoint_states = self.compute_external_boundary_states(
            t_np1=t_np1,
            dt=dt,
        )

        junction_states = self.solve_all_junctions(dt=dt)

        overlap = set(endpoint_states) & set(junction_states)
        if overlap:
            labels = sorted(endpoint.label() for endpoint in overlap)
            raise RuntimeError(
                f"Endpoint states duplicated between boundaries and junctions: {labels}."
            )

        endpoint_states.update(junction_states)

        expected = self.network.all_vessel_endpoints()
        missing = expected - set(endpoint_states)

        if missing:
            labels = sorted(endpoint.label() for endpoint in missing)
            raise RuntimeError(f"Missing endpoint states for: {labels}.")

        return endpoint_states

    def step(self, *, time: float, dt: float) -> None:
        """
        Advance all vessels by one time step.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        t_np1 = time + dt

        endpoint_states = self.compute_endpoint_states(
            t_np1=t_np1,
            dt=dt,
        )

        for vessel_id, vessel in self.network.vessels.items():
            left_endpoint, right_endpoint = self.network.endpoints_for_vessel(vessel_id)

            vessel.stepper.step(
                state_n=vessel.state_n,
                state_np1=vessel.state_np1,
                dt=dt,
                left_boundary_state=endpoint_states[left_endpoint],
                right_boundary_state=endpoint_states[right_endpoint],
            )

        for vessel in self.network.vessels.values():
            vessel.swap_states()

    def run(
        self,
        config: TimeConfig,
        record_every: int = 1,
        probes: list[ProbePoint] | None = None,
        snapshot_sample_points_by_vessel: dict[str, np.ndarray] | None = None,
        show_progress: bool = True,
        progress_description: str = "Solving network",
    ) -> NetworkSolverResult:
        """
        Run the network time loop.

        Parameters
        ----------
        config:
            Time integration configuration.

        record_every:
            Record diagnostics and probes every this many steps.

        probes:
            Optional probe points.

        snapshot_sample_points_by_vessel:
            Optional spatial sample grid per vessel. If provided, full-network
            solution snapshots are recorded at the same times as diagnostics.

        show_progress:
            If True, show a tqdm progress bar.

        progress_description:
            Text shown next to the tqdm progress bar.
        """
        if record_every <= 0:
            raise ValueError("record_every must be positive.")

        time = config.t0
        step = 0
        history = NetworkHistory()

        probe_recorder = None
        if probes is not None:
            probe_recorder = NetworkProbeRecorder(
                vessels=self.network.vessels,
                probes=probes,
            )

        snapshot_recorder = None
        if snapshot_sample_points_by_vessel is not None:
            snapshot_recorder = NetworkSnapshotRecorder(
                vessels=self.network.vessels,
                sample_points_by_vessel=snapshot_sample_points_by_vessel,
            )

        self.record_diagnostics(history, time)

        if probe_recorder is not None:
            history.probes.samples.extend(probe_recorder.sample(time))
        if snapshot_recorder is not None:
            history.snapshots.snapshots.append(snapshot_recorder.sample(time))

        # Initialize progress bar if requested
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install it with "
                    "`pip install tqdm` or `conda install -c conda-forge tqdm`."
                ) from exc

            pbar = tqdm(
                total=config.t_end - config.t0,
                desc=progress_description,
                unit="sim s",
                bar_format="{desc}: {percentage:.0f}%|{bar}| {n:.5f}/{total:.5f} [{elapsed}<{remaining}]",
            )

        try:
            while time < config.t_end:
                if step >= config.max_steps:
                    raise RuntimeError(
                        f"Reached max_steps={config.max_steps} before t_end={config.t_end}."
                    )

                dt = self.compute_dt(time=time, config=config)

                if dt <= 0.0:
                    raise RuntimeError("Computed non-positive time step.")

                self.step(time=time, dt=dt)

                time += dt
                step += 1

                if abs(time - config.t_end) <= 1.0e-14:
                    time = config.t_end

                # Update progress bar by the actual time increment
                if pbar is not None:
                    pbar.update(dt)

                if step % record_every == 0 or time >= config.t_end:
                    self.record_diagnostics(history, time)

                    if probe_recorder is not None:
                        history.probes.samples.extend(probe_recorder.sample(time))
                    if snapshot_recorder is not None:
                        history.snapshots.snapshots.append(snapshot_recorder.sample(time))
        finally:
            if pbar is not None:
                pbar.close()

        return NetworkSolverResult(
            network=self.network,
            time=time,
            num_steps=step,
            history=history,
        )
