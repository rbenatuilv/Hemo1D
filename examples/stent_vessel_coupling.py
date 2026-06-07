from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "cg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.1
DT = 1.0e-5
T_END = 1.5
POLY_ORDER = 1

RECORD_EVERY = 1
OUTPUT_DIR = None
SHOW_PLOTS = False


def main() -> None:
    model = hd.load_from_config("examples/configs/stent_vessel_coupling.json")

    q_in = hd.create_periodic_positive_sine_inflow(
        amplitude=2.0,
        duration=1.65e-1,
        period=2.5e-1,
    )

    model.set_inlet(vessel_id="upstream", kind="flowrate", function=q_in)

    model.set_windkessel_outlet(
        vessel_id="downstream",
        R_art=1.0e4,
        C=1.0e-4,
        R_ven=1.0e4,
        P_ven=0.0,
        P0=0.0,
        tissue_volume=100.0,
        bed_id="downstream_bed",
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

    model.add_probe(vessel_id="upstream", position=40.0, name="inlet")
    model.add_probe(vessel_id="stent", position=5.0, name="stent")
    model.add_probe(vessel_id="downstream", position=20.0, name="outlet")

    results = model.solve(t_end=T_END, show_progress=True)

    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}"
    else:
        name = f"method_{METHOD}"

    output_dir = OUTPUT_DIR or Path(
        f"examples/outputs/stent_vessel_coupling/{name}"
    )

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
