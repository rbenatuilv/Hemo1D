"""
Single-vessel example with a lumped capillary-bed outlet.
"""

from __future__ import annotations

from pathlib import Path

import hemo1d as hd


METHOD = "dg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.0625
DT = 1.0e-5
T_END = 5.0e-3
POLY_ORDER = 1

OUTLET_MODEL = "capillary-bed"
RECORD_EVERY = 5
OUTPUT_DIR = None
SHOW_PLOTS = False


def main() -> None:
    config_path = (
        "examples/configs/capillary_bed_outlet.json"
        if OUTLET_MODEL == "capillary-bed"
        else "examples/configs/single_vessel.json"
    )
    model = hd.load_from_config(config_path)

    q_in = hd.create_positive_sine_inflow(
        amplitude=0.005,
        duration=2.0e-3,
    )
    model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)
    if OUTLET_MODEL == "nonreflecting":
        model.set_outlet(vessel_id="vessel", kind="nonreflecting")

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
    model.add_probe(vessel_id="vessel", position=0.1 * length, name="inlet")
    model.add_probe(vessel_id="vessel", position=0.5 * length, name="mid")
    model.add_probe(vessel_id="vessel", position=length * 0.9, name="outlet")

    results = model.solve(t_end=T_END, show_progress=True)

    outlet_name = OUTLET_MODEL.replace("-", "_")
    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}_{outlet_name}"
    else:
        name = f"method_{METHOD}_{outlet_name}"

    output_dir = OUTPUT_DIR or Path(f"examples/outputs/capillary_bed_outlet/{name}")

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    bed_ids = results.capillary_bed_ids()

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    if bed_ids:
        final_bed = results.capillary_bed_history(bed_ids[0])[-1]
        print(f"Final bed pressure: {final_bed.pressure:.6e}")
        print(f"Final total inflow: {final_bed.total_inflow:.6e}")
        print(f"Final venous outflow: {final_bed.venous_outflow:.6e}")
        print(f"Final regional perfusion: {final_bed.regional_perfusion:.6e}")
    print(f"Saved outputs to {output_dir}")


def configure_shared_bed_variant(model) -> None:
    """
    Example of replacing two terminal outlets with one shared regional bed.

    This helper is not used by ``main`` because the single-vessel config has
    only one outlet. For a two-daughter terminal network, call this instead of
    ``set_windkessel_outlet`` and set ordinary inlet boundaries separately.
    """
    model.add_capillary_bed(
        bed_id="shared_region",
        outlets=[
            {"vessel_id": "daughter1", "side": "right", "R_art": 1.0e6},
            {"vessel_id": "daughter2", "side": "right", "R_art": 1.5e6},
        ],
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        P0=0.0,
        tissue_volume=100.0,
    )


if __name__ == "__main__":
    main()
