import json

import pytest

from hemo1d.config import load_network_config
from hemo1d.core.state import EndpointSide


def test_load_network_config_combined_json(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "parent": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                        "gamma_pressure_loss": 0.25,
                        "right_bound": "branch",
                        "left_bound": "inflow",
                    },
                    "d1": {
                        "length": 1.0,
                        "area0": 0.08,
                        "beta": 1.0,
                        "left_bound": "branch",
                        "right_bound": "outflow",
                    },
                    "d2": {
                        "length": 1.0,
                        "area0": 0.08,
                        "beta": 1.0,
                        "left_bound": "branch",
                        "right_bound": "outflow",
                    },
                },
                "defaults": {
                    "gamma_pressure_loss": 0.5,
                },
                "junctions": {
                    "split": {
                        "branches": ["parent", "d1", "d2"],
                        "positions": ["right", "left", "left"],
                        "angles": ["None", 0.2, 0.4],
                    }
                },
            }
        )
    )

    config = load_network_config(path)

    assert len(config.vessels) == 3
    assert len(config.junctions) == 1
    assert config.junctions[0].endpoints[0].side == EndpointSide.RIGHT
    assert config.junctions[0].angles == (None, 0.2, 0.4)
    assert config.vessels["parent"].gamma_pressure_loss == 0.25
    assert config.vessels["d1"].gamma_pressure_loss == 0.5
    assert len(config.external_endpoints()) == 3


def test_load_network_config_capillary_beds_object_form(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "vessel": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                        "left_bound": "inflow",
                        "right_bound": "outflow",
                    }
                },
                "capillary_beds": {
                    "terminal_bed": {
                        "outlets": [{"vessel_id": "vessel", "R_art": 1.0e6}],
                        "C": 1.0e-7,
                        "R_ven": 2.0e6,
                        "P_ven": 10.0,
                        "P0": 20.0,
                        "tissue_volume": 50.0,
                    }
                },
            }
        )
    )

    config = load_network_config(path)
    bed = config.capillary_beds[0]

    assert bed.bed_id == "terminal_bed"
    assert bed.compliance == pytest.approx(1.0e-7)
    assert bed.venous_resistance == pytest.approx(2.0e6)
    assert bed.venous_pressure == pytest.approx(10.0)
    assert bed.initial_pressure == pytest.approx(20.0)
    assert bed.tissue_volume == pytest.approx(50.0)
    assert bed.outlets[0].vessel_id == "vessel"
    assert bed.outlets[0].resistance == pytest.approx(1.0e6)
    assert bed.outlets[0].side is None


def test_load_network_config_capillary_beds_list_form(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "vessel": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    }
                },
                "capillary_beds": [
                    {
                        "bed_id": "terminal_bed",
                        "outlets": [
                            {
                                "vessel_id": "vessel",
                                "side": "right",
                                "R_art": 1.0e6,
                            }
                        ],
                        "C": 1.0e-7,
                        "R_ven": 2.0e6,
                        "P_ven": 10.0,
                    }
                ],
            }
        )
    )

    config = load_network_config(path)
    bed = config.capillary_beds[0]

    assert bed.bed_id == "terminal_bed"
    assert bed.initial_pressure is None
    assert bed.tissue_volume is None
    assert bed.outlets[0].side == EndpointSide.RIGHT


def test_load_network_config_two_vessel_junction(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "upstream": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    },
                    "downstream": {
                        "length": 1.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    },
                },
                "junctions": {
                    "connection": {
                        "branches": ["upstream", "downstream"],
                        "positions": ["right", "left"],
                        "angles": ["None", 0.0],
                    }
                },
            }
        )
    )

    config = load_network_config(path)

    assert len(config.vessels) == 2
    assert len(config.junctions) == 1
    assert len(config.junctions[0].endpoints) == 2
    assert config.junctions[0].endpoints[0].side == EndpointSide.RIGHT
    assert config.junctions[0].angles == (None, 0.0)
    assert len(config.external_endpoints()) == 2


def test_load_two_vessel_coupling_example_config():
    config = load_network_config("examples/configs/two_vessel_coupling.json")

    assert set(config.vessels) == {"upstream", "downstream"}
    assert len(config.junctions) == 1
    assert config.junctions[0].endpoints[0].vessel_id == "upstream"
    assert config.junctions[0].endpoints[1].vessel_id == "downstream"
    assert len(config.external_endpoints()) == 2


def test_config_rejects_unknown_junction_vessel(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "parent": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    }
                },
                "junctions": {
                    "bad_connection": {
                        "branches": ["parent", "missing1", "missing2"],
                        "positions": ["right", "left", "left"],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="unknown vessel"):
        load_network_config(path)


def test_config_rejects_obsolete_bifurcations_key(tmp_path):
    path = tmp_path / "bad_legacy.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "vessel": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    }
                },
                "bifurcations": {},
            }
        )
    )

    with pytest.raises(ValueError, match="junctions"):
        load_network_config(path)


def test_config_rejects_parent_daughter_junction_schema(tmp_path):
    path = tmp_path / "bad_parent_daughter.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "parent": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    },
                    "child": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    },
                },
                "junctions": {
                    "legacy": {
                        "parent": {"vessel_id": "parent", "side": "right"},
                        "daughter1": {"vessel_id": "child", "side": "left"},
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="branches/positions or endpoints"):
        load_network_config(path)


def test_config_rejects_capillary_beds_without_combined_config(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "vessel": {
                    "length": 2.0,
                    "area0": 0.126,
                    "beta": 1.0,
                },
                "capillary_beds": {},
            }
        )
    )

    with pytest.raises(ValueError, match="top-level 'vessels'"):
        load_network_config(path)


@pytest.mark.parametrize(
    ("capillary_beds", "message"),
    [
        (
            {
                "bad": {
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "outlets",
        ),
        (
            {
                "bad": {
                    "outlets": [],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "outlets",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "C",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "P_ven": 0.0,
                }
            },
            "R_ven",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                }
            },
            "P_ven",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": -1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "resistance",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 0.0,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "compliance",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                    "tissue_volume": -50.0,
                }
            },
            "tissue volume",
        ),
        (
            {
                "bad": {
                    "outlets": [{"vessel_id": "missing", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            },
            "unknown vessel",
        ),
        (
            [
                {
                    "id": "",
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                }
            ],
            "id must be non-empty",
        ),
        (
            [
                {
                    "id": "duplicate",
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                },
                {
                    "bed_id": "duplicate",
                    "outlets": [{"vessel_id": "vessel", "R_art": 1.0}],
                    "C": 1.0e-7,
                    "R_ven": 1.0e6,
                    "P_ven": 0.0,
                },
            ],
            "Duplicate capillary bed id",
        ),
    ],
)
def test_config_rejects_invalid_capillary_beds(tmp_path, capillary_beds, message):
    path = tmp_path / "bad_capillary_beds.json"
    path.write_text(
        json.dumps(
            {
                "vessels": {
                    "vessel": {
                        "length": 2.0,
                        "area0": 0.126,
                        "beta": 1.0,
                    }
                },
                "capillary_beds": capillary_beds,
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        load_network_config(path)
