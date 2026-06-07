from argparse import ArgumentParser


def set_arg_parser(
    description: str,
    default_method: str = "cg",
    default_h: float = 0.0625,
    default_dt: float = 1.0e-5,
    default_t_end: float = 1.0e-2,
    default_poly_order: int = 1,
    default_dg_time_scheme: str = "rk2",
    default_dg_flux: str = "lxf",
    default_outlet_model: str = "nonreflecting",
    default_record_every: int = 1,
    default_output_dir: str | None = None,
    show_plots: bool = False,
) -> ArgumentParser:
    
    parser = ArgumentParser(description=description)
    parser.add_argument("--method", choices=["cg", "dg"], default=default_method)
    parser.add_argument("--h", type=float, default=default_h)
    parser.add_argument("--dt", type=float, default=default_dt)
    parser.add_argument("--t-end", type=float, default=default_t_end)
    parser.add_argument("--poly-order", type=int, default=default_poly_order)
    parser.add_argument("--dg-time-scheme", choices=("rk2", "euler"), default=default_dg_time_scheme)
    parser.add_argument("--dg-flux", choices=("lxf", "hll"), default=default_dg_flux)
    parser.add_argument("--record-every", type=int, default=default_record_every)
    parser.add_argument("--outlet-model", choices=["nonreflecting", "capillary"], default=default_outlet_model)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--show-plots", action="store_true", default=show_plots)
    return parser
