from __future__ import annotations

import json

import pytest

import hemo1d as hd
from hemo1d.builder import SolverSettings, build_vascular_network
from hemo1d.config import load_network_config


def test_solver_settings_accepts_hll_flux() -> None:
    settings = SolverSettings(method="DG", dg_flux="hll")

    assert settings.dg_flux == "hll"


def test_solver_settings_canonicalizes_dg_flux_aliases() -> None:
    assert SolverSettings(method="DG", dg_flux="rusanov").dg_flux == "lxf"
    assert SolverSettings(method="DG", dg_flux="lax_friedrichs").dg_flux == "lxf"
    assert SolverSettings(method="DG", dg_flux="HLL").dg_flux == "hll"


def test_solver_settings_rejects_invalid_dg_flux() -> None:
    with pytest.raises(ValueError, match="Invalid DG flux scheme"):
        SolverSettings(method="DG", dg_flux="roe")


def test_model_set_solver_stores_dg_flux(tmp_path) -> None:
    model = hd.load_from_config(_write_single_vessel_config(tmp_path))

    model.set_solver(method="DG", h=0.5, dt=1.0e-5, poly_order=1, dg_flux="hll")

    assert model.solver_settings.dg_flux == "hll"


def test_builder_passes_dg_flux_to_stepper(tmp_path) -> None:
    config = load_network_config(_write_single_vessel_config(tmp_path))

    network = build_vascular_network(
        config=config,
        solver=SolverSettings(method="DG", num_cells=2, dg_flux="hll"),
        external_boundaries={},
    )

    assert network.vessels["vessel"].stepper.flux_scheme == "hll"


def _write_single_vessel_config(tmp_path):
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
