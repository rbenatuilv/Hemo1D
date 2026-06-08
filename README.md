# Hemo1D

A 1D hemodynamic network solver using Continuous Galerkin and Discontinuous
Galerkin finite element methods with FEniCSx.

## Features

- **1D hemodynamic modeling**: Solves hyperbolic conservation laws for blood flow in vessels.
- **Network support**: Handles complex vascular networks with two- and
  three-vessel junctions.
- **Continuous Galerkin (CG) and Discontinuous Galerkin (DG) discretizations**:
  Stable FEM formulations with Taylor-Galerkin CG and RK2 DG time stepping.
- **Built-in analysis tools**:
  - Richardson error estimation for convergence studies.
  - Probe sampling for monitoring simulation quantities.
  - Boundary condition compatibility equations for junctions.
- **Flexible boundary conditions**: Supports velocity, flow-rate, area, pressure, and non-reflecting boundaries.
- **Lumped outlet models**: Supports one- and multi-outlet capillary-bed /
  Windkessel coupling.
- **Clean API**: Load a configured model, attach boundaries and probes, solve, then save/plot results.

## Installation

### Prerequisites

- Python 3.10+ (`environment.yml` currently pins Python 3.12)
- [FEniCSx (dolfinx)](https://fenicsproject.org/) 0.10.x
- NumPy, SciPy, Matplotlib, pandas, tqdm

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

DG uses local Lax-Friedrichs/Rusanov flux by default (`dg_flux="lxf"`). To use
HLL instead:

```python
model.set_solver(method="DG", h=0.25, dt=1e-5, poly_order=1, dg_flux="hll")
```

See [QUICKSTART.md](QUICKSTART.md) and `examples/high_level_api.py` for fuller examples.

## Key Modules

- **`src/hemo1d/core/`**: Physics models and state representations.
- **`src/hemo1d/config/`**: Declarative vessel, junction, and capillary-bed
  configuration models plus JSON loading/validation.
- **`src/hemo1d/topology/`**: Endpoint and network graph structures.
- **`src/hemo1d/solvers/`**: Solver abstractions plus CG and DG implementations.
- **`src/hemo1d/boundary/`**: External boundary conditions and junction solvers.
- **`src/hemo1d/lumped/`**: Lumped capillary-bed outlet models.
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

Primary examples are in the `examples/` directory. They are configured with
module-level constants such as `METHOD`, `DG_FLUX`, `H`, `DT`, and `T_END`.
Run them with:

```bash
python examples/main_example.py
python examples/high_level_api.py
python examples/two_vessel_coupling.py
python examples/stent_vessel_coupling.py
python examples/aortic_endograft.py
python examples/capillary_bed_outlet.py
python examples/physiological_mca_bed_example.py
python examples/real_network.py
```

Convergence and comparison scripts live in `analysis/`:

```bash
python analysis/single_vessel.py
python analysis/three_vessel.py
python analysis/convergence_single.py
python analysis/convergence_three_vessel.py
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Bug reports and feature requests are welcome. Please use GitHub issues.
