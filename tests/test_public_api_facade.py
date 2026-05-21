import csv
import json
import subprocess
import sys

import pytest

import hemo1d as hd
from hemo1d.core.state import EndpointSide


def write_single_vessel_config(tmp_path):
    path = tmp_path / "single.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "vessel": {
                        "length": 1.0,
                        "area0": 0.126,
                        "beta": 0.060606e7,
                        "left_bound": "inflow",
                        "right_bound": "outflow",
                    }
                }
            }
        )
    )
    return path


def make_zero_flow_dg_model(tmp_path):
    model = hd.load_from_config(write_single_vessel_config(tmp_path))
    model.set_inlet(
        vessel_id="vessel",
        kind="flow_rate",
        function=lambda t: 0.0,
    )
    model.set_solver(method="DG", poly_order=1, record_every=1)
    return model


def test_public_import_does_not_eagerly_import_cg_backend():
    assert hd.load_from_config is not None
    assert not hasattr(hd, "NetworkSolver")
    assert not hasattr(hd, "VascularNetwork")
    assert not hasattr(hd, "ProbePoint")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, hemo1d; print('dolfinx' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_load_from_config_and_solve_single_vessel_dg(tmp_path):
    model = hd.load_from_config(write_single_vessel_config(tmp_path))

    model.set_inlet(
        vessel_id="vessel",
        kind="flow_rate",
        function=lambda t: 0.0,
    )
    model.set_solver(method="DG", h=0.5, dt=1.0e-5, poly_order=1)
    model.add_probe(vessel_id="vessel", position=0.5)

    result = model.solve(t_end=1.0e-5, show_progress=False)

    assert isinstance(result, hd.Results)
    assert result.time == pytest.approx(1.0e-5)
    assert result.num_steps == 1
    assert len(result.history.probes.samples) == 2

    output_dir = tmp_path / "result"
    result.save(output_dir)
    assert (output_dir / "probes.csv").exists()
    assert (output_dir / "diagnostics.csv").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "final_states" / "vessel_final_state.csv").exists()


def test_endpoint_side_can_be_inferred_from_config_label(tmp_path):
    model = hd.load_from_config(write_single_vessel_config(tmp_path))

    model.set_inlet(
        vessel_id="vessel",
        kind="velocity",
        function=lambda t: 0.0,
    )
    model.set_outlet(vessel_id="vessel")

    assert any(endpoint.side == EndpointSide.LEFT for endpoint in model._boundaries)


def test_missing_boundary_is_reported(tmp_path):
    model = hd.load_from_config(write_single_vessel_config(tmp_path))
    model.set_solver(method="DG", h=0.5, dt=1.0e-5, poly_order=1)

    with pytest.raises(ValueError, match="Missing external boundary"):
        model.solve(t_end=1.0e-5, show_progress=False)


def test_convergence_uses_full_solution_without_probes(tmp_path):
    model = make_zero_flow_dg_model(tmp_path)

    study = model.convergence_test(
        h_levels=[0.5, 0.25, 0.125],
        dt_levels=[1.0e-5, 5.0e-6, 2.5e-6],
        expected_order=2.0,
        t_end=1.0e-5,
        show_progress=False,
    )

    rows = study.error_rows

    assert len(rows) == 2
    assert [row.level_name for row in rows] == ["L0", "L1"]
    assert [row.num_cells for row in rows] == [2, 4]
    assert [row.h_like for row in rows] == pytest.approx([0.5, 0.25])
    assert set(study.observed_orders) == {"area", "flow_rate"}
    assert len(study.observed_orders["area"]) == 1
    assert len(study.observed_orders["flow_rate"]) == 1

    output_dir = tmp_path / "convergence"
    study.save(output_dir)

    with (output_dir / "convergence.csv").open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == [
            "level_name",
            "num_cells",
            "dt",
            "h_like",
            "area_error",
            "flow_rate_error",
        ]
        assert len(list(reader)) == 2


def test_convergence_errors_do_not_depend_on_probes(tmp_path):
    no_probe_model = make_zero_flow_dg_model(tmp_path)
    with_probe_model = make_zero_flow_dg_model(tmp_path)
    with_probe_model.add_probe(vessel_id="vessel", position=0.5)

    kwargs = {
        "h_levels": [0.5, 0.25, 0.125],
        "dt_levels": [1.0e-5, 5.0e-6, 2.5e-6],
        "expected_order": 2.0,
        "t_end": 1.0e-5,
        "show_progress": False,
    }

    no_probe_rows = no_probe_model.convergence_test(**kwargs).error_rows
    with_probe_rows = with_probe_model.convergence_test(**kwargs).error_rows

    assert [
        (row.level_name, row.area_error, row.flow_rate_error)
        for row in with_probe_rows
    ] == [
        (row.level_name, row.area_error, row.flow_rate_error)
        for row in no_probe_rows
    ]


def test_convergence_requires_constant_decreasing_h_levels(tmp_path):
    model = make_zero_flow_dg_model(tmp_path)

    with pytest.raises(ValueError, match="strictly decreasing"):
        model.convergence_test(
            h_levels=[0.25, 0.5],
            dt_levels=[1.0e-5, 5.0e-6],
            expected_order=2.0,
            t_end=1.0e-5,
            show_progress=False,
        )

    with pytest.raises(ValueError, match="constant refinement ratio"):
        model.convergence_test(
            h_levels=[0.5, 0.25, 0.2],
            dt_levels=[1.0e-5, 5.0e-6, 2.5e-6],
            expected_order=2.0,
            t_end=1.0e-5,
            show_progress=False,
        )


def test_convergence_progress_reports_current_level(tmp_path, capsys):
    model = make_zero_flow_dg_model(tmp_path)

    model.convergence_test(
        h_levels=[0.5, 0.25],
        dt_levels=[1.0e-5, 5.0e-6],
        expected_order=2.0,
        t_end=1.0e-5,
        show_progress=True,
    )

    captured = capsys.readouterr()

    assert (
        "Convergence L0 (1/2): method=dg, poly_order=1, "
        "h=0.5, dt=1e-05, t_end=1e-05"
    ) in captured.out
    assert (
        "Convergence L1 (2/2): method=dg, poly_order=1, "
        "h=0.25, dt=5e-06, t_end=1e-05"
    ) in captured.out
