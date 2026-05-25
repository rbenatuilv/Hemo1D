from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import hemo1d as hd


# =============================================================================
# Experiment 2: Left ICA stenosis sweep with pressure inlets
# =============================================================================

METHOD = "CG"
DG_TIME_SCHEME = "rk2"

DEGREE = 1
H = 0.0625
DT = 1.0e-5
T_END = 2.0

RECORD_EVERY = 10
SHOW_PLOTS = False

BASE_CONFIG_PATH = Path("experiments/configs/CoW_normal.json")
GENERATED_CONFIG_DIR = Path("experiments/configs/generated_stenosis")
OUTPUT_ROOT = Path("experiments/outputs/CoW_LICA_stenosis_pressure_sweep")

MMHG_TO_DYN_CM2 = 1333.22

# Original area of L-ICA_I in your CoW config.
L_ICA_BASE_AREA = 0.1252

# Area-stenosis surrogate:
# 0.70 means the area is reduced by 70%, so A_new = 0.30 * A_original.
STENOSIS_LEVELS = {
    "normal": 0.00,
    "LICA_area30": 0.30,
    "LICA_area50": 0.50,
    "LICA_area70": 0.70,
    "LICA_area85": 0.85,
}

# Expected healthy targets from Experiment 1.
EXPECTED_BEDS = {
    "L-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0},
    "R-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0},
    "L-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0},
    "R-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0},
    "L-PCA_bed": {"target_flow_ml_min": 85.0, "target_perfusion": 50.0},
    "R-PCA_bed": {"target_flow_ml_min": 85.0, "target_perfusion": 50.0},
}

# Probe the main inlet and communicating vessels.
# For ACA: positive flow is from the left ACA side to the right ACA side.
# For L-PCommA/R-PCommA: positive flow is from PCA side to ICA side.
PROBE_VESSELS = (
    "BAS",
    "L-ICA_I",
    "R-ICA_I",
    "ACA",
    "L-PCommA",
    "R-PCommA",
    "L-MCA",
    "R-MCA",
    "L-ACA_II",
    "R-ACA_II",
    "L-PCA_II",
    "R-PCA_II",
)


# =============================================================================
# Pressure inlet waveform
# =============================================================================

HEART_RATE = 1.2      # Hz, about 72 bpm
RAMP_TIME = 0.20      # seconds


def mmhg(value: float) -> float:
    return value * MMHG_TO_DYN_CM2


def ramp(t: float) -> float:
    """Smooth enough startup for the pulsatile part."""
    return min(1.0, max(0.0, t / RAMP_TIME))


def arterial_pressure(
    t: float,
    *,
    mean_mmhg: float,
    amp1_mmhg: float,
    amp2_mmhg: float,
    phase_s: float = 0.0,
) -> float:
    """
    Simple physiological pressure waveform.

    Mean pressure is always present.
    Pulsatility is gradually ramped in to avoid a violent startup transient.
    """
    tau = t - phase_s
    omega = 2.0 * np.pi * HEART_RATE

    pulsatile_mmhg = (
        amp1_mmhg * np.sin(omega * tau)
        + amp2_mmhg * np.sin(2.0 * omega * tau - 0.8)
    )

    pressure_mmhg = mean_mmhg + ramp(t) * pulsatile_mmhg
    return mmhg(pressure_mmhg)


# =============================================================================
# Config generation
# =============================================================================

def make_stenosis_config(case_name: str, area_stenosis: float) -> Path:
    """
    Create a generated JSON config for one stenosis level.

    This keeps the base config unchanged and writes a modified copy.
    """
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with BASE_CONFIG_PATH.open() as f:
        config = json.load(f)

    config = copy.deepcopy(config)

    # Put the resting vessel reference pressure near 80 mmHg.
    # This makes the initial vessel state closer to the capillary-bed calibration.
    config.setdefault("defaults", {})
    config["defaults"]["p0"] = mmhg(80.0)
    config["defaults"]["p_ext"] = 0.0

    remaining_area_fraction = 1.0 - area_stenosis
    stenosed_area = L_ICA_BASE_AREA * remaining_area_fraction

    config["vessels"]["L-ICA_I"]["initial_area"] = stenosed_area

    path = GENERATED_CONFIG_DIR / f"CoW_{case_name}.json"
    with path.open("w") as f:
        json.dump(config, f, indent=2)

    return path


# =============================================================================
# Model setup
# =============================================================================

def configure_model(config_path: Path):
    model = hd.load_from_config(config_path)

    # Pressure inlets.
    # With pressure inlets, the flow is not imposed directly.
    # The flow emerges from the pressure sources + CoW geometry + 0D beds.
    model.set_inlet(
        vessel_id="L-ICA_I",
        kind="pressure",
        function=lambda t: arterial_pressure(
            t,
            mean_mmhg=85.0,
            amp1_mmhg=14.0,
            amp2_mmhg=4.0,
            phase_s=0.00,
        ),
    )

    model.set_inlet(
        vessel_id="R-ICA_I",
        kind="pressure",
        function=lambda t: arterial_pressure(
            t,
            mean_mmhg=85.0,
            amp1_mmhg=14.0,
            amp2_mmhg=4.0,
            phase_s=0.00,
        ),
    )

    model.set_inlet(
        vessel_id="BAS",
        kind="pressure",
        function=lambda t: arterial_pressure(
            t,
            mean_mmhg=84.0,
            amp1_mmhg=11.0,
            amp2_mmhg=3.0,
            phase_s=0.04,
        ),
    )

    model.set_solver(
        method=METHOD,
        h=H,
        dt=DT,
        poly_order=DEGREE,
        dg_time_scheme=DG_TIME_SCHEME,
        record_every=RECORD_EVERY,
    )

    for vessel_id in PROBE_VESSELS:
        if vessel_id not in model.config.vessels:
            continue

        length = model.config.vessel(vessel_id).length
        model.add_probe(vessel_id=vessel_id, position=0.5 * length, name="mid")

    return model


# =============================================================================
# Diagnostics
# =============================================================================

def late_mask(times: np.ndarray) -> np.ndarray:
    """Use the final 25% of recorded samples."""
    tail_start = times[-1] - 0.25 * (times[-1] - times[0])
    return times >= tail_start


def bed_late_means(results, bed_id: str) -> dict[str, float]:
    samples = results.capillary_bed_history(bed_id)

    times = np.array([s.time for s in samples])
    tail = late_mask(times)

    pcap_mmhg = np.array([s.pressure for s in samples]) / MMHG_TO_DYN_CM2
    inflow_ml_min = np.array([s.total_inflow for s in samples]) * 60.0
    venous_ml_min = np.array([s.venous_outflow for s in samples]) * 60.0
    perfusion_ml_100g_min = np.array([s.regional_perfusion for s in samples]) * 6000.0

    return {
        "pcap_mmhg": float(np.mean(pcap_mmhg[tail])),
        "flow_ml_min": float(np.mean(inflow_ml_min[tail])),
        "venous_ml_min": float(np.mean(venous_ml_min[tail])),
        "perfusion": float(np.mean(perfusion_ml_100g_min[tail])),
    }


def probe_late_mean_flow(results, vessel_id: str, probe_name: str = "mid") -> float | None:
    samples = results.history.probes.by_vessel_and_name(vessel_id, probe_name)
    if not samples:
        return None

    times = np.array([s.time for s in samples])
    tail = late_mask(times)

    flow_ml_min = np.array([s.flow_rate for s in samples]) * 60.0
    return float(np.mean(flow_ml_min[tail]))


def print_case_summary(
    case_name: str,
    area_stenosis: float,
    results,
) -> dict[str, dict[str, float]]:

    print(f"\n=== {case_name} ===")
    print(f"L-ICA_I area stenosis: {100.0 * area_stenosis:.0f}%")
    print(f"L-ICA_I A0 used:       {L_ICA_BASE_AREA * (1.0 - area_stenosis):.5f} cm^2\n")

    case = {}

    header = (
        f"{'Bed':<12}"
        f"{'Pcap':>10}"
        f"{'Flow':>12}"
        f"{'Perfusion':>12}"
        f"{'Healthy target':>16}"
    )
    print(header)
    print("-" * len(header))

    for bed_id, expected in EXPECTED_BEDS.items():
        means = bed_late_means(results, bed_id)
        case[bed_id] = means

        print(
            f"{bed_id:<12}"
            f"{means['pcap_mmhg']:>10.2f}"
            f"{means['flow_ml_min']:>12.1f}"
            f"{means['perfusion']:>12.1f}"
            f"{expected['target_perfusion']:>16.1f}"
        )

    total_flow = sum(case[bed_id]["flow_ml_min"] for bed_id in EXPECTED_BEDS)

    print("-" * len(header))
    print(f"{'TOTAL':<12}{'':>10}{total_flow:>12.1f}")

    print("\nMean mid-vessel flows, late time:")
    for vessel_id in ("L-ICA_I", "R-ICA_I", "BAS", "ACA", "L-PCommA", "R-PCommA"):
        q = probe_late_mean_flow(results, vessel_id)
        if q is None:
            continue
        print(f"  {vessel_id:<8}: {q:>9.2f} mL/min")

    print("\nFlow sign reminders:")
    print("  ACA positive      = left ACA side -> right ACA side")
    print("  PCommA positive   = PCA side -> ICA side")

    return case


def print_preservation_table(all_cases: dict[str, dict[str, dict[str, float]]]) -> None:
    normal = all_cases["normal"]

    print("\n=== Perfusion preservation ratio vs normal ===")
    print("1.00 means unchanged relative to the normal simulated case.\n")

    header = f"{'Case':<14}" + "".join(f"{bed_id:<12}" for bed_id in EXPECTED_BEDS)
    print(header)
    print("-" * len(header))

    for case_name, case in all_cases.items():
        row = f"{case_name:<14}"
        for bed_id in EXPECTED_BEDS:
            ratio = case[bed_id]["perfusion"] / normal[bed_id]["perfusion"]
            row += f"{ratio:<12.2f}"
        print(row)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    all_cases = {}

    for case_name, area_stenosis in STENOSIS_LEVELS.items():
        config_path = make_stenosis_config(case_name, area_stenosis)

        model = configure_model(config_path)
        results = model.solve(t_end=T_END, show_progress=True)

        output_dir = OUTPUT_ROOT / case_name
        results.save(output_dir)
        results.plot_capillary_beds(output_dir / "plots", show=SHOW_PLOTS)

        case = print_case_summary(case_name, area_stenosis, results)
        all_cases[case_name] = case

        print(f"\nSaved outputs to {output_dir}")

    print_preservation_table(all_cases)


if __name__ == "__main__":
    main()