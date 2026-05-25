from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import hemo1d as hd


METHOD = "CG"
DG_TIME_SCHEME = "rk2"

DEGREE = 1
H = 0.0625
DT = 1e-5
T_END = 2.0

RECORD_EVERY = 10
SHOW_PLOTS = False

CONFIG_PATH = Path("experiments/configs/CoW_normal.json")
INFLOWS_DIR = Path("data/inflows")
OUTPUT_DIR = Path("experiments/outputs/CoW_normal")


MMHG_TO_DYN_CM2 = 1333.22

EXPECTED_BEDS = {
    "L-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
    "R-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
    "L-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
    "R-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
    "L-PCA_bed": {"target_flow_ml_min": 85.0,  "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
    "R-PCA_bed": {"target_flow_ml_min": 85.0,  "target_perfusion": 50.0, "target_pcap_mmhg": 35.0},
}


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


def summarize_capillary_beds(results) -> None:
    print("\n=== Expected vs simulated regional perfusion ===")
    print("Late-time means are computed over the last 25% of recorded samples.\n")

    total_expected_flow = 0.0
    total_simulated_flow = 0.0

    header = (
        f"{'Bed':<12}"
        f"{'Pcap exp':>10} {'Pcap sim':>10}"
        f"{'Flow exp':>12} {'Flow sim':>12}"
        f"{'Perf exp':>12} {'Perf sim':>12}"
    )
    print(header)
    print("-" * len(header))

    for bed_id, expected in EXPECTED_BEDS.items():
        samples = results.capillary_bed_history(bed_id)

        times = np.array([sample.time for sample in samples])
        pcap = np.array([sample.pressure for sample in samples]) / MMHG_TO_DYN_CM2
        inflow = np.array([sample.total_inflow for sample in samples]) * 60.0
        venous = np.array([sample.venous_outflow for sample in samples]) * 60.0
        perfusion = np.array([sample.regional_perfusion for sample in samples]) * 6000.0

        tail_start = times[-1] - 0.25 * (times[-1] - times[0])
        tail = times >= tail_start

        mean_pcap = float(np.mean(pcap[tail]))
        mean_inflow = float(np.mean(inflow[tail]))
        mean_venous = float(np.mean(venous[tail]))
        mean_perf = float(np.mean(perfusion[tail]))

        total_expected_flow += expected["target_flow_ml_min"]
        total_simulated_flow += mean_inflow

        print(
            f"{bed_id:<12}"
            f"{expected['target_pcap_mmhg']:>10.2f} {mean_pcap:>10.2f}"
            f"{expected['target_flow_ml_min']:>12.1f} {mean_inflow:>12.1f}"
            f"{expected['target_perfusion']:>12.1f} {mean_perf:>12.1f}"
        )

        print(
            f"{'':<12}"
            f"{'Pcap range':>10} {float(np.min(pcap[tail])):>4.1f}–{float(np.max(pcap[tail])):<4.1f} mmHg"
            f"   venous out: {mean_venous:.1f} mL/min"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<12}{'':>20}{total_expected_flow:>12.1f} {total_simulated_flow:>12.1f}")
    print("\nAcceptance target:")
    print("  perfusion: 45–55 mL/100g/min in all six beds")
    print("  Pcap:      roughly 30–40 mmHg")
    print("  total flow close to 750 mL/min")


def main() -> None:

    model = hd.load_from_config(CONFIG_PATH)

    for vessel_id in ("BAS", "L-ICA_I", "R-ICA_I"):
        velocity = hd.read_velocity_csv(
            INFLOWS_DIR / f"{vessel_id}.csv",
            out_of_bounds="periodic",
            ramp_time=0.1,
        )
        model.set_inlet(vessel_id=vessel_id, kind="velocity", function=velocity)

    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=DEGREE,
        dg_time_scheme=DG_TIME_SCHEME,
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

    output_dir = OUTPUT_DIR
    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    summarize_capillary_beds(results)

    print(f"\nFinished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()