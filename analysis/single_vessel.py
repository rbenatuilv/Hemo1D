from __future__ import annotations
from pathlib import Path

import hemo1d as hd
from _cli import set_arg_parser
from _capillary_outlets import CAPILLARY_BED_PARAMS


def main() -> None:
    parser = set_arg_parser(
        description="Single-vessel simulation",

        default_method="cg",

        default_dg_time_scheme="rk2",
        default_dg_flux="lxf",

        default_h=0.0625,
        default_dt=1.0e-5,
        default_t_end=1.0e-2,
        default_poly_order=1,

        default_outlet_model="capillary",

        default_record_every=5,
    )

    args = parser.parse_args()


    model = hd.load_from_config("analysis/configs/single_vessel.json")

    q_in = hd.create_positive_sine_inflow(
        amplitude=0.005,
        duration=2.0e-3,
    )
    model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)

    if args.outlet_model == "capillary":
        model.set_windkessel_outlet(
        vessel_id="vessel",
        bed_id="terminal_bed",
        C=CAPILLARY_BED_PARAMS["C"],
        R_art=CAPILLARY_BED_PARAMS["R_art"],
        R_ven=CAPILLARY_BED_PARAMS["R_ven"],
        P_ven=CAPILLARY_BED_PARAMS["P_ven"],
        P0=CAPILLARY_BED_PARAMS["P0"],
        tissue_volume=CAPILLARY_BED_PARAMS["tissue_volume"],
    )
    
    model.set_solver(
        method=args.method,
        h=args.h,
        dt=args.dt,
        poly_order=args.poly_order,
        dg_time_scheme=args.dg_time_scheme,
        dg_flux=args.dg_flux,
        record_every=args.record_every,
    )

    for name, position in (("x_0-25cm", 0.25), ("x_1cm", 1.0), ("x_1-75cm", 1.75)):
        model.add_probe(vessel_id="vessel", position=position, name=name)

    results = model.solve(t_end=args.t_end, show_progress=True)


    if args.method == "dg":
        name = f"method_{args.method}_{args.dg_time_scheme}_{args.dg_flux}_{args.outlet_model}"
    else:
        name = f"method_{args.method}_{args.outlet_model}"

    output_dir = args.output_dir or Path(f"analysis/outputs/single_vessel/{name}")

    results.save(output_dir)

    results.plot_probes(output_dir / "plots", show=args.show_plots)

    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=args.show_plots)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")

if __name__ == "__main__":
    main()
