from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np

from hemo1d.convergence.reports import ConvergenceErrorRow


def _plot_single_field_convergence(
    h: np.ndarray,
    errors: np.ndarray,
    field_label: str,
    output_path: str | Path | None,
    title: str,
    reference_orders: tuple[float, ...],
    show: bool,
) -> None:
    """
    Plot one convergence curve with nearby reference slopes.

    The reference slopes are anchored at the first error point of the same
    field, so they appear close to the corresponding error graph.
    """
    import matplotlib.pyplot as plt

    if np.any(h <= 0.0):
        raise ValueError("All h_like values must be positive.")
    if np.any(errors <= 0.0):
        raise ValueError("All errors must be positive for log-log plotting.")

    fig, ax = plt.subplots()

    ax.loglog(h, errors, "o-", label=field_label)

    # Anchor reference slopes at the first point of this specific field.
    h_anchor = h[0]
    error_anchor = errors[0]

    h_ref = np.array([np.min(h), np.max(h)], dtype=float)

    for p in reference_orders:
        ref = error_anchor * (h_ref / h_anchor) ** p
        ax.loglog(h_ref, ref, "--", label=f"slope p={p:g}")

    ax.invert_xaxis()
    ax.set_xlabel("h-like quantity = 1 / N")
    ax.set_ylabel(r"$L^\infty_t(L^2_z)$ error")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence_errors_separate(
    rows: Sequence[ConvergenceErrorRow],
    output_dir: str | Path | None = None,
    filename_prefix: str = "convergence",
    title_prefix: str = "Convergence study",
    reference_orders: tuple[float, ...] = (0.5, 1.0, 2.0),
    show: bool = True,
) -> None:
    """
    Plot area and flow-rate convergence errors separately.

    Generates:
        {filename_prefix}_area.png
        {filename_prefix}_flow_rate.png

    if output_dir is provided.
    """
    if len(rows) == 0:
        raise ValueError("Cannot plot convergence with no error rows.")

    h = np.array([row.h_like for row in rows], dtype=float)
    area_errors = np.array([row.area_error for row in rows], dtype=float)
    flow_errors = np.array([row.flow_rate_error for row in rows], dtype=float)

    area_path = None
    flow_path = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        area_path = output_dir / f"{filename_prefix}_area.png"
        flow_path = output_dir / f"{filename_prefix}_flow_rate.png"

    _plot_single_field_convergence(
        h=h,
        errors=area_errors,
        field_label="Area error",
        output_path=area_path,
        title=f"{title_prefix}: area",
        reference_orders=reference_orders,
        show=show,
    )

    _plot_single_field_convergence(
        h=h,
        errors=flow_errors,
        field_label="Flow-rate error",
        output_path=flow_path,
        title=f"{title_prefix}: flow rate",
        reference_orders=reference_orders,
        show=show,
    )