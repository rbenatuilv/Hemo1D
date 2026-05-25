from __future__ import annotations

import copy
import json
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

BASE_CONFIG_PATH = Path("experiments/configs/CoW_normal.json")
GENERATED_CONFIG_DIR = Path("experiments/configs/generated_stenosis")
INFLOWS_DIR = Path("data/inflows")
OUTPUT_ROOT = Path("experiments/outputs/CoW_LICA_stenosis_sweep")

MMHG_TO_DYN_CM2 = 1333.22

L_ICA_BASE_AREA = 0.1252

STENOSIS_LEVELS = {
    "normal": 0.00,
    "LICA_area30": 0.30,
    "LICA_area50": 0.50,
    "LICA_area70": 0.70,
    "LICA_area85": 0.85,
}

EXPECTED_BEDS = {
    "L-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0},
    "R-ACA_bed": {"target_flow_ml_min": 100.0, "target_perfusion": 50.0},
    "L-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0},
    "R-MCA_bed": {"target_flow_ml_min": 190.0, "target_perfusion": 50.0},
    "L-PCA_bed": {"target_flow_ml_min": 85.0, "target_perfusion": 50.0},
    "R-PCA_bed": {"target_flow_ml_min": 85.0, "target_perfusion": 50.0},
}


def make_stenosis_config(case_name: str, area_stenosis: float) -> Path:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with BASE_CONFIG_PATH.open() as f:
        config = json.load(f)

    config = copy.deepcopy(config)

    remaining_area_fraction = 1.0 - area_stenosis
    stenosed_area = L_ICA_BASE_AREA * remaining_area_fraction

    config["vessels"]["L-ICA_I"]["initial_area"] = stenosed_area

    path = GENERATED_CONFIG_DIR / f"CoW_{case_name}.json"
    with path.open("w") as f:
        json.dump(config, f, indent=2)

    return path


def configure_model(config_path: Path):
    model = hd.load_from_config(config_path)

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

    return model


def bed_late_means(results, bed_id: str) -> dict[str, float]:
    samples = results.capillary_bed_history(bed_id)

    times = np.array([s.time for s in samples])
    pcap = np.array([s.pressure for s in samples]) / MMHG_TO_DYN_CM2
    inflow = np.array([s.total_inflow for s in samples]) * 60.0
    venous = np.array([s.venous_outflow for s in samples]) * 60.0
    perfusion = np.array([s.regional_perfusion for s in samples]) * 6000.0

    tail_start = times[-1] - 0.25 * (times[-1] - times[0])
    tail = times >= tail_start

    return {
        "pcap_mmhg": float(np.mean(pcap[tail])),
        "flow_ml_min": float(np.mean(inflow[tail])),
        "venous_ml_min": float(np.mean(venous[tail])),
        "perfusion": float(np.mean(perfusion[tail])),
    }


def print_case_summary(case_name: str, area_stenosis: float, results) -> dict[str, dict[str, float]]:
    print(f"\n=== {case_name} ===")
    print(f"L-ICA_I area stenosis: {100.0 * area_stenosis:.0f}%")
    print(f"L-ICA_I A0 used:       {L_ICA_BASE_AREA * (1.0 - area_stenosis):.5f} cm^2\n")

    case = {}

    header = (
        f"{'Bed':<12}"
        f"{'Pcap':>10}"
        f"{'Flow':>12}"
        f"{'Perfusion':>12}"
        f"{'Expected':>12}"
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
            f"{expected['target_perfusion']:>12.1f}"
        )

    total_flow = sum(case[bed_id]["flow_ml_min"] for bed_id in EXPECTED_BEDS)
    print("-" * len(header))
    print(f"{'TOTAL':<12}{'':>10}{total_flow:>12.1f}")

    return case


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

        print(f"Saved outputs to {output_dir}")

    print("\n=== Perfusion preservation ratio vs normal ===")
    normal = all_cases["normal"]

    header = f"{'Case':<14}" + "".join(f"{bed_id:<12}" for bed_id in EXPECTED_BEDS)
    print(header)
    print("-" * len(header))

    for case_name, case in all_cases.items():
        row = f"{case_name:<14}"
        for bed_id in EXPECTED_BEDS:
            ratio = case[bed_id]["perfusion"] / normal[bed_id]["perfusion"]
            row += f"{ratio:<12.2f}"
        print(row)


if __name__ == "__main__":
    main()