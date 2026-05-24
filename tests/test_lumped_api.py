import json

import numpy as np
import pytest

import hemo1d as hd
from hemo1d.core.state import EndpointSide
from hemo1d.topology.endpoint import NetworkEndpoint


def test_capillary_bed_import_compatibility():
    from hemo1d.lumped import (
        CapillaryBedEndpoint as PackageEndpoint,
        CapillaryBedSample as PackageSample,
        LumpedCapillaryBed as PackageBed,
    )
    from hemo1d.lumped.capillary_bed import (
        CapillaryBedEndpoint as ModuleEndpoint,
        CapillaryBedSample as ModuleSample,
        LumpedCapillaryBed as ModuleBed,
    )

    assert PackageEndpoint is ModuleEndpoint
    assert PackageSample is ModuleSample
    assert PackageBed is ModuleBed
    assert ModuleBed.__module__ == "hemo1d.lumped.capillary_bed"


def write_two_vessel_config(tmp_path):
    path = tmp_path / "two_outlets.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "v1": {
                        "length": 1.0,
                        "area0": 0.126,
                        "beta": 0.060606e7,
                        "left_bound": "inflow",
                        "right_bound": "outflow",
                    },
                    "v2": {
                        "length": 1.0,
                        "area0": 0.126,
                        "beta": 0.060606e7,
                        "left_bound": "inflow",
                        "right_bound": "outflow",
                    },
                }
            }
        )
    )
    return path


def test_set_windkessel_outlet_replaces_default_outlet(tmp_path):
    model = hd.load_from_config(write_two_vessel_config(tmp_path))
    endpoint = NetworkEndpoint("v1", EndpointSide.RIGHT)

    assert endpoint in model._boundaries

    model.set_windkessel_outlet(
        vessel_id="v1",
        R_art=1.0e6,
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        side="right",
        bed_id="v1_bed",
    )

    assert endpoint not in model._boundaries
    assert any(endpoint in bed.endpoint_set() for bed in model._lumped_beds)

    with pytest.raises(ValueError, match="lumped capillary bed"):
        model.set_outlet(vessel_id="v1", side="right")


def test_add_capillary_bed_replaces_default_outlets(tmp_path):
    model = hd.load_from_config(write_two_vessel_config(tmp_path))
    endpoint_1 = NetworkEndpoint("v1", EndpointSide.RIGHT)
    endpoint_2 = NetworkEndpoint("v2", EndpointSide.RIGHT)

    model.add_capillary_bed(
        bed_id="shared",
        outlets=[
            {"vessel_id": "v1", "side": "right", "R_art": 1.0e6},
            ("v2", 2.0e6, "right"),
        ],
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
    )

    assert endpoint_1 not in model._boundaries
    assert endpoint_2 not in model._boundaries
    assert model._lumped_beds[0].endpoint_set() == {endpoint_1, endpoint_2}


def test_capillary_bed_diagnostics_and_result_accessors(tmp_path):
    model = hd.load_from_config(write_two_vessel_config(tmp_path))
    model.set_inlet(vessel_id="v1", kind="flow_rate", function=lambda t: 0.0)
    model.set_inlet(vessel_id="v2", kind="flow_rate", function=lambda t: 0.0)
    model.set_windkessel_outlet(
        vessel_id="v1",
        R_art=1.0e6,
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        tissue_volume=50.0,
        side="right",
        bed_id="v1_bed",
    )
    model.set_solver(method="DG", h=0.5, dt=1.0e-5, poly_order=1, record_every=1)

    result = model.solve(t_end=1.0e-5, show_progress=False)

    samples = result.capillary_bed_history("v1_bed")
    pressures = result.capillary_bed_pressure("v1_bed")
    perfusion = result.regional_perfusion("v1_bed")

    assert len(samples) == 2
    assert pressures.shape == (2,)
    assert perfusion.shape == (2,)
    assert np.all(np.isfinite(pressures))
    assert np.all(np.isfinite(perfusion))
    assert "v1_bed" in result.raw.history.diagnostics[-1].lumped_bed_samples
    assert samples[0].endpoint_inflows
    assert samples[-1].total_inflow == pytest.approx(
        sum(samples[-1].endpoint_inflows.values())
    )


def test_plot_capillary_bed_histories(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    model = hd.load_from_config(write_two_vessel_config(tmp_path))
    model.set_inlet(vessel_id="v1", kind="flow_rate", function=lambda t: 0.0)
    model.set_inlet(vessel_id="v2", kind="flow_rate", function=lambda t: 0.0)
    model.set_windkessel_outlet(
        vessel_id="v1",
        R_art=1.0e6,
        C=1.0e-7,
        R_ven=1.0e6,
        P_ven=0.0,
        tissue_volume=50.0,
        side="right",
        bed_id="v1_bed",
    )
    model.set_solver(method="DG", h=0.5, dt=1.0e-5, poly_order=1, record_every=1)

    result = model.solve(t_end=1.0e-5, show_progress=False)
    plot_dir = tmp_path / "plots"

    assert result.capillary_bed_ids() == ["v1_bed"]

    result.plot_capillary_beds(plot_dir, show=False)

    assert (plot_dir / "v1_bed_capillary_bed_pressure_history.png").exists()
    assert (plot_dir / "v1_bed_capillary_bed_flow_history.png").exists()
    assert (
        plot_dir / "v1_bed_capillary_bed_regional_perfusion_history.png"
    ).exists()
