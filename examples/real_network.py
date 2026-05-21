from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


SELECTED_PROBE_VESSELS = (
    "BAS",
    "L-PCA_I",
    "L-PCA_II",
    "L-PCommA",
    "L-ICA_I",
    "L-ICA_II",
    "L-ACA_I",
    "L-MCA",
    "R-PCA_I",
    "R-PCA_II",
    "R-PCommA",
    "R-ICA_I",
    "R-ICA_II",
    "R-ACA_I",
    "R-MCA",
    "ACA",
    "L-ACA_II",
    "R-ACA_II",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real network simulation")
    parser.add_argument("--method", choices=("cg", "dg"), default="cg")
    parser.add_argument("--h", type=float, default=0.125)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--dg-time-scheme", choices=("rk2", "euler"), default="rk2")
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("data/network.json")

    for vessel_id in ("BAS", "L-ICA_I", "R-ICA_I"):
        velocity = hd.read_velocity_csv(
            Path("data/inflows") / f"{vessel_id}.csv",
            out_of_bounds="periodic",
            ramp_time=0.1,
        )
        model.set_inlet(vessel_id=vessel_id, kind="velocity", function=velocity)

    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=args.poly_order,
        dg_time_scheme=args.dg_time_scheme,
        record_every=args.record_every,
    )

    for vessel_id in SELECTED_PROBE_VESSELS:
        if vessel_id not in model.config.vessels:
            continue
        length = model.config.vessel(vessel_id).length
        model.add_probe(vessel_id=vessel_id, position=0.0, name="left")
        model.add_probe(vessel_id=vessel_id, position=0.5 * length, name="mid")
        model.add_probe(vessel_id=vessel_id, position=length, name="right")

    results = model.solve(t_end=args.t_end, show_progress=True)

    output_dir = args.output_dir or Path(f"examples/outputs/real_network_{args.method}")
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
