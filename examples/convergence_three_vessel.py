from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-vessel convergence study")
    parser.add_argument("--method", choices=("cg", "dg"), default="dg")
    parser.add_argument("--poly-order", type=int, default=1)
    parser.add_argument("--expected-order", type=float, default=2.0)
    parser.add_argument("--t-end", type=float, default=5.0e-4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("examples/configs/three_vessel.json")
    model.set_inlet(
        vessel_id="parent",
        kind="flow_rate",
        function=hd.create_positive_sine_inflow(amplitude=1.0e-3, duration=5.0e-4),
    )
    model.set_outlet(vessel_id="daughter1")
    model.set_outlet(vessel_id="daughter2")
    model.set_solver(method=args.method, poly_order=args.poly_order, record_every=1)

    for vessel_id in ("parent", "daughter1", "daughter2"):
        length = model.config.vessel(vessel_id).length
        model.add_probe(vessel_id=vessel_id, position=0.5 * length, name="mid")

    study = model.convergence_test(
        h_levels=[0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
        dt_levels=[4.0e-6, 2.0e-6, 1.0e-6, 0.5e-6, 0.25e-6],
        expected_order=args.expected_order,
        t_end=args.t_end,
        show_progress=True,
    )

    output_dir = args.output_dir or Path(f"examples/outputs/convergence_three_vessel_{args.method}")
    study.save(output_dir)
    study.plot(output_dir, show=args.show_plots)

    print("Observed full-solution orders:", study.observed_orders)
    print(f"Saved convergence outputs to {output_dir}")


if __name__ == "__main__":
    main()
