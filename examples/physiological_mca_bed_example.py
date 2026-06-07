"""
Physiological sanity test for one MCA-like regional capillary bed.

Assumed units used by Hemo1D examples:
    length: cm
    area: cm^2
    flow: cm^3/s = mL/s
    pressure: dyn/cm^2

Physiological target:
    one MCA-like region, tissue volume ≈ 300 mL / 300 g
    target regional CBF ≈ 50 mL / 100 g / min
    target mean regional flow = 2.5 mL/s = 150 mL/min
    terminal arterial pressure ≈ 80 mmHg
    capillary-bed pressure ≈ 35 mmHg
    venous pressure ≈ 8 mmHg
"""
from __future__ import annotations

import math
from pathlib import Path

import hemo1d as hd

MMHG_TO_DYN_CM2 = 1333.22


def dyn_to_mmhg(value: float) -> float:
    return value / MMHG_TO_DYN_CM2


METHOD = "dg"
DG_FLUX = "lxf"
DG_TIME_SCHEME = "rk2"

H = 0.20
DT = 2.0e-5
T_END = 1.5
POLY_ORDER = 1

RECORD_EVERY = 250
OUTPUT_DIR = None
SHOW_PLOTS = False

# Target mean perfusion: 50 mL/100g/min over a 300 g region.
TISSUE_VOLUME = 300.0        # cm^3 ≈ g, assuming density ≈ 1 g/mL
Q_MEAN = 2.5                 # cm^3/s = 150 mL/min
HEART_RATE = 1.0             # Hz = 60 bpm
PULSATILITY = 0.25           # +/-25% around mean, always positive


def inlet_flow(t: float) -> float:
    return Q_MEAN * (1.0 + PULSATILITY * math.sin(2.0 * math.pi * HEART_RATE * t))


def perfusion_ml_100g_min(flow_ml_s: float, tissue_volume_ml: float) -> float:
    # flow / volume gives 1/s. Multiply by 60 s/min and 100 g.
    return flow_ml_s / tissue_volume_ml * 6000.0


def main() -> None:
    config_path = Path("examples/configs/physiological_mca_bed_config.json")
    model = hd.load_from_config(config_path)

    model.set_inlet(
        vessel_id="left_mca",
        kind="flow_rate",
        function=inlet_flow,
        side="left",
    )

    # dt may need adjustment depending on your mesh/order and local wave speed.
    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=POLY_ORDER,
        dg_time_scheme=DG_TIME_SCHEME,
        dg_flux=DG_FLUX,
        record_every=RECORD_EVERY,
    )

    length = model.config.vessel("left_mca").length
    model.add_probe(vessel_id="left_mca", position=0.5 * length, name="mid_mca")
    model.add_probe(vessel_id="left_mca", position=0.95 * length, name="distal_mca")

    results = model.solve(t_end=T_END, show_progress=True)

    if METHOD == "dg":
        name = f"method_{METHOD}_{DG_FLUX}"
    else:
        name = f"method_{METHOD}"

    output_dir = OUTPUT_DIR or Path(
        f"examples/outputs/physiological_mca_bed/{name}"
    )

    results.save(output_dir)
    results.plot_probes(output_dir / "plots", show=SHOW_PLOTS)
    if results.capillary_bed_ids():
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

    bed = results.capillary_bed_history("L_MCA_region")

    # Use the second half of the simulation to skip the initial transient.
    tail = bed[len(bed) // 2:]
    mean_pcap = sum(s.pressure for s in tail) / len(tail)
    mean_qin = sum(s.total_inflow for s in tail) / len(tail)
    mean_qven = sum(s.venous_outflow for s in tail) / len(tail)
    mean_perf = perfusion_ml_100g_min(mean_qin, TISSUE_VOLUME)

    print("\n=== Expected physiological sanity check ===")
    print("Target P_cap:       35 mmHg")
    print("Target mean inflow: 2.5 mL/s = 150 mL/min")
    print("Target perfusion:   50 mL/100g/min")
    print("\n=== Simulated late-time means ===")
    print(f"Mean P_cap:       {dyn_to_mmhg(mean_pcap):.2f} mmHg")
    print(f"Mean inflow:      {mean_qin:.3f} mL/s = {60.0 * mean_qin:.1f} mL/min")
    print(f"Mean venous out:  {mean_qven:.3f} mL/s = {60.0 * mean_qven:.1f} mL/min")
    print(f"Mean perfusion:   {mean_perf:.1f} mL/100g/min")
    min_pcap = min(s.pressure for s in tail)
    max_pcap = max(s.pressure for s in tail)
    print(
        f"P_cap range tail: {dyn_to_mmhg(min_pcap):.2f}–"
        f"{dyn_to_mmhg(max_pcap):.2f} mmHg"
    )
    print(f"\nFinished at t={results.time:.6e} with {results.num_steps} steps")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
