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
                "bifurcations": {
                    "bif": {
                        "branches": ["parent", "d1", "d2"],
                        "positions": ["right", "left", "left"],
                    }
                },
            }
        )
    )

    config = load_network_config(path)

    assert len(config.vessels) == 3
    assert len(config.bifurcations) == 1
    assert config.bifurcations[0].parent.side == EndpointSide.RIGHT
    assert len(config.external_endpoints()) == 3


def test_config_rejects_unknown_bifurcation_vessel(tmp_path):
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
                "bifurcations": {
                    "bif": {
                        "branches": ["parent", "missing1", "missing2"],
                        "positions": ["right", "left", "left"],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="unknown vessel"):
        load_network_config(path)
