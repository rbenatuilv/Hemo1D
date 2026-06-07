from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "dg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.0275
DT = 1.0e-5
T_END = 0.3
POLY_ORDER = 1

OUTLET_MODEL = "nonreflecting"
RECORD_EVERY = 1
OUTPUT_DIR = None
SHOW_PLOTS = False


def main() -> None:
    model = hd.load_from_config("examples/configs/aortic_endograft.json")

    q_in = hd.create_periodic_positive_sine_inflow(
        amplitude=40.0,
        duration=1.0e-1,
        period=1.0e-1,
    )

    model.set_inlet(vessel_id="upstream", kind="flowrate", function=q_in)

    AREA_1 = model.config.vessel("branch_1").area0
    AREA_2 = model.config.vessel("branch_2").area0

    if OUTLET_MODEL == "nonreflecting":
        model.set_outlet(vessel_id="branch_1", kind="nonreflecting")
        model.set_outlet(vessel_id="branch_2", kind="nonreflecting")
    else:
        model.set_windkessel_outlet(
            vessel_id="branch_1",
            R_art=1.0e3,
            C=1.0e-3,
            R_ven=1.0e3,
            P_ven=0.0,
            P0=0.0,
            tissue_volume=100.0 * AREA_1,
            bed_id="branch_1_bed",
        )
        model.set_windkessel_outlet(
            vessel_id="branch_2",
            R_art=1.0e3,
            C=1.0e-3,
            R_ven=1.0e3,
            P_ven=0.0,
            P0=0.0,
            tissue_volume=100.0 * AREA_2,
            bed_id="branch_2_bed",
        )

    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=POLY_ORDER,
        dg_time_scheme=DG_TIME_SCHEME,
        dg_flux=DG_FLUX,
        record_every=RECORD_EVERY,
    )

    LENGTH = model.config.vessel("upstream").length

    model.add_probe(vessel_id="upstream", position=LENGTH * 0.5, name="inlet")
    model.add_probe(vessel_id="branch_1", position=LENGTH * 0.5, name="outlet_1")
    model.add_probe(vessel_id="branch_2", position=LENGTH * 0.5, name="outlet_2")

    results = model.solve(t_end=T_END, show_progress=True)

    outlet_name = OUTLET_MODEL.replace("-", "_")
    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}_{outlet_name}"
    else:
        name = f"method_{METHOD}_{outlet_name}"

    output_dir = OUTPUT_DIR or Path(f"examples/outputs/aortic_endograft/{name}")

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
