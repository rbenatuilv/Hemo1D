from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-vessel junction coupling simulation")
    parser.add_argument("--method", choices=("cg", "dg"), default="cg")
    parser.add_argument("--h", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--t-end", type=float, default=1.5)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--dg-time-scheme", choices=("rk2", "euler"), default="rk2")
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("examples/configs/two_vessel_coupling.json")

    q_in = hd.create_periodic_positive_sine_inflow(
        amplitude=2.0,
        duration=1.65e-1,
        period=2.5e-1,
    )

    model.set_inlet(vessel_id="upstream", kind="flowrate", function=q_in)
    model.set_outlet(vessel_id="downstream", kind="nonreflecting")

    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=args.poly_order,
        dg_time_scheme=args.dg_time_scheme,
        record_every=args.record_every,
    )

    model.add_probe(vessel_id="upstream", position=20, name="inlet")
    model.add_probe(vessel_id="downstream", position=20, name="outlet")

    results = model.solve(t_end=args.t_end, show_progress=True)

    output_dir = args.output_dir or Path(f"examples/outputs/two_vessel_coupling_{args.method}")
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
