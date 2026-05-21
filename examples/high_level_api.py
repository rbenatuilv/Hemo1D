from __future__ import annotations

import argparse
from pathlib import Path

import hemo1d as hd


def main() -> None:
    parser = argparse.ArgumentParser(description="High-level real-network facade example")
    parser.add_argument("--method", choices=("cg", "dg"), default="dg")
    parser.add_argument("--h", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=1.0e-5)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/outputs/high_level_api"))
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    model = hd.load_from_config("data/network.json")

    bas_velocity = hd.read_velocity_csv(
        "data/inflows/BAS.csv",
        out_of_bounds="periodic",
        ramp_time=0.1,
    )
    l_ica_velocity = hd.read_velocity_csv(
        "data/inflows/L-ICA_I.csv",
        out_of_bounds="periodic",
        ramp_time=0.1,
    )
    r_ica_velocity = hd.read_velocity_csv(
        "data/inflows/R-ICA_I.csv",
        out_of_bounds="periodic",
        ramp_time=0.1,
    )

    model.set_inlet(vessel_id="BAS", kind="velocity", function=bas_velocity)
    model.set_inlet(vessel_id="L-ICA_I", kind="velocity", function=l_ica_velocity)
    model.set_inlet(vessel_id="R-ICA_I", kind="velocity", function=r_ica_velocity)

    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=1,
    )

    model.add_probe(vessel_id="BAS", position=0.5, name="bas_mid")
    model.add_probe(vessel_id="L-MCA", position=2.0, name="l_mca_mid")

    results = model.solve(t_end=args.t_end, show_progress=True)

    output_dir = args.output_dir
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} in {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
