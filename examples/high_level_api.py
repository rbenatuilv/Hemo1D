from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "dg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.25
DT = 1.0e-5
T_END = 1.0
POLY_ORDER = 1

OUTLET_MODEL = "nonreflecting"
RECORD_EVERY = 1
OUTPUT_DIR = None
SHOW_PLOTS = False


def main() -> None:
    config_path = (
        "examples/configs/CoW_normal.json"
        if OUTLET_MODEL == "capillary-bed"
        else "data/network.json"
    )
    model = hd.load_from_config(config_path)

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
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=POLY_ORDER,
        dg_time_scheme=DG_TIME_SCHEME,
        dg_flux=DG_FLUX,
        record_every=RECORD_EVERY,
    )

    model.add_probe(vessel_id="BAS", position=0.5, name="bas_mid")
    model.add_probe(vessel_id="L-MCA", position=2.0, name="l_mca_mid")

    results = model.solve(t_end=T_END, show_progress=True)

    outlet_name = OUTLET_MODEL.replace("-", "_")
    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}_{outlet_name}"
    else:
        name = f"method_{METHOD}_{outlet_name}"

    output_dir = OUTPUT_DIR or Path(f"examples/outputs/high_level_api/{name}")

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} in {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
