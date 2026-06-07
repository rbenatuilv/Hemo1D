"""
Minimal Hemo1D example using the public facade.
"""

from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "dg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.125
DT = 1.0e-5
T_END = 2.0e-3
POLY_ORDER = 1

OUTLET_MODEL = "nonreflecting"
RECORD_EVERY = 10
OUTPUT_DIR = None
SHOW_PLOTS = False


def main() -> None:
    model = hd.load_from_config("examples/configs/single_vessel.json")

    q_in = hd.create_pulsatile_inflow(
        systolic_amplitude=0.005,
        systolic_duration=0.35,
        cycle_period=1.0,
    )
    model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)
    if OUTLET_MODEL == "nonreflecting":
        model.set_outlet(vessel_id="vessel")
    else:
        model.set_windkessel_outlet(
            vessel_id="vessel",
            R_art=1.0e6,
            C=1.0e-7,
            R_ven=1.0e6,
            P_ven=0.0,
            P0=0.0,
            tissue_volume=50.0,
            bed_id="terminal_bed",
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

    length = model.config.vessel("vessel").length
    model.add_probe(vessel_id="vessel", position=0.0, name="inlet")
    model.add_probe(vessel_id="vessel", position=0.5 * length, name="mid")
    model.add_probe(vessel_id="vessel", position=length, name="outlet")

    results = model.solve(t_end=T_END, show_progress=True)

    outlet_name = OUTLET_MODEL.replace("-", "_")
    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}_{outlet_name}"
    else:
        name = f"method_{METHOD}_{outlet_name}"

    output_dir = OUTPUT_DIR or Path(f"examples/outputs/main_example/{name}")

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
