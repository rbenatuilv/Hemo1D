from __future__ import annotations

from pathlib import Path

import hemo1d as hd

from _capillary_outlets import CAPILLARY_BED_PARAMS
from _cli import set_arg_parser


def main() -> None:
    EXPECTED_ORDER = 2.0

    parser = set_arg_parser(
        description="Single-vessel convergence",

        default_method="cg",

        default_dg_time_scheme="rk2",
        default_dg_flux="lxf",

        default_t_end=5.0e-4,
        default_poly_order=1,

        default_outlet_model="capillary",

        default_record_every=5,
        show_plots=True,
    )
    args = parser.parse_args()

    model = hd.load_from_config("analysis/configs/single_vessel.json")
    model.set_inlet(
        vessel_id="vessel",
        kind="flow_rate",
        function=hd.create_positive_sine_inflow(amplitude=1.0e-3, duration=5.0e-4),
    )
    
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
        poly_order=args.poly_order,
        dg_flux=args.dg_flux,
        record_every=args.record_every,
    )

    study = model.convergence_test(
        h_levels=[0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
        dt_levels=[4.0e-6, 2.0e-6, 1.0e-6, 0.5e-6, 0.25e-6],
        expected_order=EXPECTED_ORDER,
        t_end=args.t_end,
        show_progress=True,
    )

    if args.method == "dg":
        name = f"method_{args.method}_{args.dg_time_scheme}_{args.dg_flux}_{args.outlet_model}"
    else:
        name = f"method_{args.method}_{args.outlet_model}"

    output_dir = args.output_dir or Path(f"analysis/outputs/convergence_single/{name}")
    study.save(output_dir)
    study.plot(output_dir, show=args.show_plots)

    print("Observed full-solution orders:", study.observed_orders)
    print(f"Saved convergence outputs to {output_dir}")


if __name__ == "__main__":
    main()
