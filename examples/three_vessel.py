from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-vessel bifurcation simulation")
    parser.add_argument("--method", choices=("cg", "dg"), default="cg")
    parser.add_argument("--h", type=float, default=0.03125)
    parser.add_argument("--dt", type=float, default=2.0e-6)
    parser.add_argument("--t-end", type=float, default=5.0e-3)
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--dg-time-scheme", choices=("rk2", "euler"), default="rk2")
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("examples/configs/three_vessel.json")

    q_in = hd.create_positive_sine_inflow(
        amplitude=1.0e-3,
        duration=5.0e-4,
    )
    model.set_inlet(vessel_id="parent", kind="flow_rate", function=q_in)
    model.set_outlet(vessel_id="daughter1", kind="nonreflecting")
    model.set_outlet(vessel_id="daughter2", kind="nonreflecting")

    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=args.poly_order,
        dg_time_scheme=args.dg_time_scheme,
        record_every=args.record_every,
    )

    for vessel_id in ("parent", "daughter1", "daughter2"):
        length = model.config.vessel(vessel_id).length
        model.add_probe(vessel_id=vessel_id, position=0.0, name="left")
        model.add_probe(vessel_id=vessel_id, position=0.5 * length, name="mid")
        model.add_probe(vessel_id=vessel_id, position=length, name="right")

    results = model.solve(t_end=args.t_end, show_progress=True)

    output_dir = args.output_dir or Path(f"examples/outputs/three_vessel_{args.method}")
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
