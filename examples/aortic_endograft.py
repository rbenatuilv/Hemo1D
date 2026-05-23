from __future__ import annotations

from pathlib import Path

import hemo1d as hd


def main() -> None:

    METHOD = "dg"
    H = 0.0275
    DT = 1.0e-5
    POLY_ORDER = 1
    DG_TIME_SCHEME = "rk2"
    RECORD_EVERY = 1

    T_END = 0.3
    OUTPUT_DIR = Path(f"examples/outputs/aortic_endograft_{METHOD}")
    SHOW_PLOTS = True
    

    model = hd.load_from_config("examples/configs/aortic_endograft.json")

    q_in = hd.create_periodic_positive_sine_inflow(
        amplitude=40.0,
        duration=1.0e-1,
        period=1.0e-1,
    )

    model.set_inlet(vessel_id="upstream", kind="flowrate", function=q_in)

    AREA_1 = model.config.vessel("branch_1").area0
    AREA_2 = model.config.vessel("branch_2").area0

    model.set_outlet(vessel_id="branch_1", kind="nonreflecting")
    model.set_outlet(vessel_id="branch_2", kind="nonreflecting")

    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=POLY_ORDER,
        dg_time_scheme=DG_TIME_SCHEME,
        record_every=RECORD_EVERY,
    )

    LENGTH = model.config.vessel("upstream").length

    model.add_probe(vessel_id="upstream", position=LENGTH * 0.5, name="inlet")
    model.add_probe(vessel_id="branch_1", position=LENGTH * 0.5, name="outlet_1")
    model.add_probe(vessel_id="branch_2", position=LENGTH * 0.5, name="outlet_2")

    results = model.solve(t_end=T_END, show_progress=True)

    output_dir = OUTPUT_DIR
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)

    print(f"Finished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
