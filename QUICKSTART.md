# Hemo1D Quick Start

Use Hemo1D through the high-level facade:

```python
import hemo1d as hd

model = hd.load_from_config("examples/configs/single_vessel.json")

q_in = hd.create_positive_sine_inflow(amplitude=0.005, duration=2.0e-3)
model.set_inlet(vessel_id="vessel", kind="flow_rate", function=q_in)
model.set_outlet(vessel_id="vessel", kind="nonreflecting")

model.set_solver(method="DG", h=0.125, dt=1.0e-5, poly_order=1)

model.add_probe(vessel_id="vessel", position=1.0, name="x_1cm")
model.add_probe(vessel_id="vessel", position=2.5, name="mid")

results = model.solve(t_end=8.0e-3)
results.save("examples/outputs/single_vessel_dg")
results.plot_probes("examples/outputs/single_vessel_dg/plots")
```

DG defaults to local Lax-Friedrichs/Rusanov interface flux (`dg_flux="lxf"`).
Select HLL with:

```python
model.set_solver(method="DG", h=0.125, dt=1.0e-5, poly_order=1, dg_flux="hll")
```

## Config Files

Network JSON files contain vessels plus optional junctions:

```json
{
  "vessels": {
    "upstream": {
      "length": 2.0,
      "area0": 0.126,
      "beta": 606060.0,
      "left_bound": "inflow",
      "right_bound": "coupling"
    }
  },
  "junctions": {}
}
```

`data/network.json` is the canonical 18-vessel example network. Minimal example
configs live in `examples/configs/`.

## CSV Boundary Data

```python
velocity = hd.read_velocity_csv(
    "data/inflows/BAS.csv",
    out_of_bounds="periodic",
    ramp_time=0.1,
)
model.set_inlet(vessel_id="BAS", kind="velocity", function=velocity)
```

CSV readers support `out_of_bounds="error"`, `"constant"`, and `"periodic"`,
with optional ramp-up to avoid a sharp start at `t=0`.

## Convergence Studies

Convergence studies report full-network `L∞(time; L2(space))` Richardson errors
for area and flow rate.

```python
study = model.convergence_test(
    h_levels=[0.2, 0.1, 0.05],
    dt_levels=[1.0e-4, 5.0e-5, 2.5e-5],
    expected_order=2,
    t_end=1.0e-3,
)

study.save("examples/outputs/convergence")
study.plot("examples/outputs/convergence")
print(study.observed_orders)
```

## Examples

```bash
python examples/main_example.py
python examples/single_vessel.py --method dg
python examples/stent_vessel_coupling.py --method dg
python examples/three_vessel.py --method dg
python examples/real_network.py --method cg
python examples/convergence_single.py --method dg
python examples/convergence_three_vessel.py --method dg
```

CG runs require the FEniCSx environment from `environment.yml`:

```bash
conda run -n hemo1d-fenicsx python -m pytest -q
```

The public `hemo1d` namespace intentionally exposes the facade and common
configuration, boundary, I/O, result, convergence, and core physics types.
Low-level solver work should import from canonical subpackages such as
`hemo1d.solvers.cg`, `hemo1d.solvers.dg`, `hemo1d.topology`,
`hemo1d.boundary`, and `hemo1d.observe`.
