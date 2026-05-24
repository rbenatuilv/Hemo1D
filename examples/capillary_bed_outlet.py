"""
Single-vessel example with a lumped capillary-bed outlet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-vessel capillary-bed outlet example"
    )
    parser.add_argument("--method", choices=("cg", "dg"), default="dg")
    parser.add_argument("--h", type=float, default=0.0625)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--t-end", type=float, default=5.0e-3)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--dg-time-scheme", choices=("rk2", "euler"), default="rk2")
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/capillary_bed_outlet"),
    )
    args = parser.parse_args()

    model = hd.load_from_config("examples/configs/capillary_bed_outlet.json")

    q_in = hd.create_positive_sine_inflow(
        amplitude=0.005,
        duration=2.0e-3,
    )
    model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)

    model.set_windkessel_outlet(
        vessel_id="vessel",
        side="right",
        bed_id="terminal_bed",
        R_art=1.0e6,
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        P0=0.0,
        tissue_volume=50.0,
    )

    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=args.poly_order,
        record_every=args.record_every,
    )

    length = model.config.vessel("vessel").length
    model.add_probe(vessel_id="vessel", position=0.1 * length, name="inlet")
    model.add_probe(vessel_id="vessel", position=0.5 * length, name="mid")
    model.add_probe(vessel_id="vessel", position=length*0.9, name="outlet")

    results = model.solve(t_end=args.t_end, show_progress=True)
    results.save(args.output_dir)
    results.plot_probes(args.output_dir / "plots", show=args.show_plots)
    results.plot_capillary_beds(args.output_dir / "plots", show=args.show_plots)

    bed_samples = results.capillary_bed_history("terminal_bed")
    final_bed = bed_samples[-1]

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Final bed pressure: {final_bed.pressure:.6e}")
    print(f"Final total inflow: {final_bed.total_inflow:.6e}")
    print(f"Final venous outflow: {final_bed.venous_outflow:.6e}")
    print(f"Final regional perfusion: {final_bed.regional_perfusion:.6e}")
    print(f"Saved outputs to {args.output_dir}")


def configure_shared_bed_variant(model) -> None:
    """
    Example of replacing two terminal outlets with one shared regional bed.

    This helper is not used by ``main`` because the single-vessel config has
    only one outlet. For a two-daughter terminal network, call this instead of
    ``set_windkessel_outlet`` and set ordinary inlet boundaries separately.
    """
    model.add_capillary_bed(
        bed_id="shared_region",
        outlets=[
            {"vessel_id": "daughter1", "side": "right", "R_art": 1.0e6},
            {"vessel_id": "daughter2", "side": "right", "R_art": 1.5e6},
        ],
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        P0=0.0,
        tissue_volume=100.0,
    )


if __name__ == "__main__":
    main()
