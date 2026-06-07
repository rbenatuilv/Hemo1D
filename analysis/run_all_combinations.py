#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


SCRIPTS = (
    "single_vessel",
    "three_vessel",
    "convergence_single",
    "convergence_three_vessel",
)

SOLVERS = (
    ("cg", ("--method", "cg")),
    ("dg_lxf", ("--method", "dg", "--dg-flux", "lxf")),
    ("dg_hll", ("--method", "dg", "--dg-flux", "hll")),
)

OUTLETS = (
    ("nonreflecting", "nonreflecting"),
    ("capillary_bed", "capillary-bed"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all analysis combinations.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use tiny t_end values.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/outputs/all_combinations"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    jobs = build_jobs(repo_root, output_root, smoke=args.smoke)

    if args.dry_run:
        for command, _ in jobs:
            print(" ".join(str(part) for part in command))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_job, command, output_dir, env)
            for command, output_dir in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            command, output_dir, returncode = future.result()
            label = output_dir.relative_to(output_root)
            if returncode == 0:
                print(f"OK     {label}")
            else:
                failures += 1
                print(f"FAIL   {label}  see {output_dir / 'run.log'}")
                print("       " + " ".join(str(part) for part in command))

    return 1 if failures else 0


def build_jobs(
    repo_root: Path,
    output_root: Path,
    *,
    smoke: bool,
) -> list[tuple[list[str], Path]]:
    jobs = []
    for script in SCRIPTS:
        for solver_name, solver_args in SOLVERS:
            for outlet_name, outlet_arg in OUTLETS:
                output_dir = output_root / script / solver_name / outlet_name
                command = [
                    sys.executable,
                    str(repo_root / "analysis" / f"{script}.py"),
                    *solver_args,
                    "--outlet-model",
                    outlet_arg,
                    "--output-dir",
                    str(output_dir),
                ]
                if smoke:
                    t_end = "1.0e-6" if script.startswith("convergence") else "1.0e-5"
                    command.extend(["--t-end", t_end])
                jobs.append((command, output_dir))
    return jobs


def run_job(
    command: list[str],
    output_dir: Path,
    env: dict[str, str],
) -> tuple[list[str], Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w") as log:
        log.write("$ " + " ".join(str(part) for part in command) + "\n\n")
        log.flush()
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return command, output_dir, process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
