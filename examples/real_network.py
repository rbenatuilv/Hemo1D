from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "cg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.125
DT = 1.0e-5
T_END = 1.0
POLY_ORDER = 1

OUTLET_MODEL = "nonreflecting"
RECORD_EVERY = 10
OUTPUT_DIR = None
SHOW_PLOTS = False

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
    config_path = (
        "examples/configs/CoW_normal.json"
        if OUTLET_MODEL == "capillary-bed"
        else "examples/configs/CoW_no_capillaries.json"
    )
    model = hd.load_from_config(config_path)

    for vessel_id in ("BAS", "L-ICA_I", "R-ICA_I"):
        velocity = hd.read_velocity_csv(
            Path("data/inflows") / f"{vessel_id}.csv",
            out_of_bounds="periodic",
            ramp_time=0.1,
        )
        model.set_inlet(vessel_id=vessel_id, kind="velocity", function=velocity)

    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=POLY_ORDER,
        dg_time_scheme=DG_TIME_SCHEME,
        dg_flux=DG_FLUX,
        record_every=RECORD_EVERY,
    )

    for vessel_id in SELECTED_PROBE_VESSELS:
        if vessel_id not in model.config.vessels:
            continue
        length = model.config.vessel(vessel_id).length
        model.add_probe(vessel_id=vessel_id, position=0.0, name="left")
        model.add_probe(vessel_id=vessel_id, position=0.5 * length, name="mid")
        model.add_probe(vessel_id=vessel_id, position=length, name="right")

    results = model.solve(t_end=T_END, show_progress=True)

    outlet_name = OUTLET_MODEL.replace("-", "_")
    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}_{outlet_name}"
    else:
        name = f"method_{METHOD}_{outlet_name}"

    output_dir = OUTPUT_DIR or Path(f"examples/outputs/real_network/{name}")

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
