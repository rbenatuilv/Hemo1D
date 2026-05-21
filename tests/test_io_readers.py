import pytest

from hemo1d.io import read_velocity_csv


def test_csv_reader_constant_out_of_bounds(tmp_path):
    path = tmp_path / "inlet.csv"
    path.write_text("Time,Velocity\n0.0,1.0\n1.0,3.0\n")

    fn = read_velocity_csv(path, out_of_bounds="constant")

    assert fn(-1.0) == pytest.approx(1.0)
    assert fn(0.5) == pytest.approx(2.0)
    assert fn(2.0) == pytest.approx(3.0)


def test_csv_reader_periodic_out_of_bounds(tmp_path):
    path = tmp_path / "inlet.csv"
    path.write_text("Time,Velocity\n0.0,1.0\n1.0,3.0\n")

    fn = read_velocity_csv(path, out_of_bounds="periodic")

    assert fn(1.5) == pytest.approx(2.0)


def test_csv_reader_ramp(tmp_path):
    path = tmp_path / "inlet.csv"
    path.write_text("Time,Velocity\n0.0,2.0\n1.0,2.0\n")

    fn = read_velocity_csv(
        path,
        out_of_bounds="constant",
        ramp_time=1.0,
        ramp_kind="linear",
    )

    assert fn(0.0) == pytest.approx(0.0)
    assert fn(0.5) == pytest.approx(1.0)
    assert fn(1.0) == pytest.approx(2.0)
