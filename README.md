# Hemo1D

A 1D hemodynamic network solver using Continuous Galerkin and Discontinuous Galerkin finite element methods with FEniCSx. TEST

## Features

- **1D hemodynamic modeling**: Solves hyperbolic conservation laws for blood flow in vessels.
- **Network support**: Handles complex vascular networks with two- and three-vessel junctions.
- **Continuous Galerkin (CG) and Discontinuous Galerkin (DG) discretizations**: Stable, accurate FEM formulations with Taylor-Galerkin and Lax-Friedrichs time stepping.
- **Built-in analysis tools**:
  - Richardson error estimation for convergence studies.
  - Probe sampling for monitoring simulation quantities.
  - Boundary condition compatibility equations for junctions.
- **Flexible boundary conditions**: Supports velocity, flow-rate, area, pressure, and non-reflecting boundaries.
- **Clean API**: Load a configured model, attach boundaries and probes, solve, then save/plot results.

## Installation

### Prerequisites

- Python 3.10+
- [FEniCSx (dolfinx)](https://fenicsproject.org/) >= 0.6.0
- NumPy, Matplotlib, tqdm

### Setup

1. **Using Conda** (recommended):

```bash
conda env create -f environment.yml
conda activate hemo1d-fenicsx
pip install -e .
```

2. **Manual installation**:

```bash
pip install .
```

## Quick Start

The high-level API is available directly from `import hemo1d as hd`:

```python
import hemo1d as hd

model = hd.load_from_config("data/network.json")

v_in = hd.read_velocity_csv(
    "data/inflows/BAS.csv",
    out_of_bounds="periodic",
    ramp_time=0.1,
)

model.set_inlet(vessel_id="BAS", kind="velocity", function=v_in)
model.set_solver(method="DG", h=0.25, dt=1e-5, poly_order=1)
model.add_probe(vessel_id="BAS", position=0.5)

results = model.solve(t_end=1e-3)
results.save("outputs/high_level_api")
results.plot_probes("outputs/high_level_api/plots")
```

See [QUICKSTART.md](QUICKSTART.md) and `examples/high_level_api.py` for fuller examples.

## Key Modules

- **`src/hemo1d/core/`**: Physics models and state representations.
- **`src/hemo1d/solvers/`**: Solver abstractions plus CG and DG implementations.
- **`src/hemo1d/topology/`**: Endpoint and network graph structures.
- **`src/hemo1d/boundary/`**: External boundary conditions and junction solvers.
- **`src/hemo1d/observe/`**: Probe sampling, histories, and diagnostics.
- **`src/hemo1d/convergence/`**: Full-solution error estimation and verification tools.
- **`src/hemo1d/io/`**: Config readers, CSV boundary input readers, and result writers.

## Architecture

For detailed design and API documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing

Run the test suite:

```bash
conda run -n hemo1d-fenicsx python -m pytest -q
```

Pure Python/DG subsets can run outside that environment, but CG tests require
the FEniCSx/PETSc/NumPy stack from `environment.yml`.

## Examples

All examples are in the `examples/` directory. Run them with:

```bash
python examples/main_example.py
python examples/single_vessel.py
python examples/stent_vessel_coupling.py
python examples/three_vessel.py
python examples/convergence_single.py
python examples/convergence_three_vessel.py
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Bug reports and feature requests are welcome. Please use GitHub issues.
