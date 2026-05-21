from hemo1d.convergence.errors import (
    l2_error_1d,
    l2_norm_history,
    linf_time_l2_space_error,
    observed_orders,
    relative_error,
)
from hemo1d.convergence.network_errors import (
    compute_network_richardson_error_rows,
    linf_time_network_l2_error,
    network_l2_error_at_time,
    network_observed_orders,
    richardson_extrapolate_network_history,
    richardson_extrapolate_network_snapshot,
)
from hemo1d.convergence.network_snapshots import (
    NetworkSnapshotHistory,
    NetworkSnapshotRecorder,
    NetworkSolutionSnapshot,
)
from hemo1d.convergence.plotting import plot_convergence_errors_separate
from hemo1d.convergence.reports import (
    ConvergenceErrorRow,
    ConvergenceLevel,
    compute_errors_against_reference,
    compute_richardson_error_rows,
    write_convergence_errors_csv,
)
from hemo1d.convergence.richardson import (
    richardson_extrapolate_history,
    richardson_extrapolate_snapshot,
    richardson_extrapolate_values,
)
from hemo1d.convergence.snapshots import (
    SnapshotHistory,
    SolutionSnapshot,
    VesselSnapshotRecorder,
)


def __getattr__(name: str):
    if name in {"ConvergenceStudy", "ConvergenceStudyLevel"}:
        from hemo1d.api import ConvergenceStudy, ConvergenceStudyLevel

        return {
            "ConvergenceStudy": ConvergenceStudy,
            "ConvergenceStudyLevel": ConvergenceStudyLevel,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConvergenceStudy",
    "ConvergenceStudyLevel",
    "ConvergenceErrorRow",
    "ConvergenceLevel",
    "NetworkSnapshotHistory",
    "NetworkSnapshotRecorder",
    "NetworkSolutionSnapshot",
    "SnapshotHistory",
    "SolutionSnapshot",
    "VesselSnapshotRecorder",
    "compute_errors_against_reference",
    "compute_network_richardson_error_rows",
    "compute_richardson_error_rows",
    "l2_error_1d",
    "l2_norm_history",
    "linf_time_l2_space_error",
    "linf_time_network_l2_error",
    "network_l2_error_at_time",
    "network_observed_orders",
    "observed_orders",
    "plot_convergence_errors_separate",
    "relative_error",
    "richardson_extrapolate_history",
    "richardson_extrapolate_network_history",
    "richardson_extrapolate_network_snapshot",
    "richardson_extrapolate_snapshot",
    "richardson_extrapolate_values",
    "write_convergence_errors_csv",
]
