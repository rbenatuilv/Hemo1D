"""
Minimal Hemo1D example using the public facade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Hemo1D facade example")
    parser.add_argument("--method", choices=("cg", "dg"), default="dg")
    parser.add_argument("--h", type=float, default=0.125)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--t-end", type=float, default=2.0e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/outputs/main_example"))
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("examples/configs/single_vessel.json")

    q_in = hd.create_pulsatile_inflow(
        systolic_amplitude=0.005,
        systolic_duration=0.35,
        cycle_period=1.0,
    )
    model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)
    model.set_outlet(vessel_id="vessel")

    model.set_solver(method=args.method, h=args.h, dt=args.dt, poly_order=1, record_every=10)

    length = model.config.vessel("vessel").length
    model.add_probe(vessel_id="vessel", position=0.0, name="inlet")
    model.add_probe(vessel_id="vessel", position=0.5 * length, name="mid")
    model.add_probe(vessel_id="vessel", position=length, name="outlet")

    results = model.solve(t_end=args.t_end, show_progress=True)

    output_dir = args.output_dir
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
