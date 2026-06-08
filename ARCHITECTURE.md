# Hemo1D Architecture

Hemo1D is organized around a small public facade and canonical internal
subpackages. Normal users should import the package as:

```python
import hemo1d as hd
```

and work with `hd.load_from_config(...)`, `HemodynamicModel`, and `Results`.

## Public Facade

- `hemo1d.__init__`: public API only.
- `hemo1d.api`: `HemodynamicModel`, `NetworkModel`, `load_from_config`, and the
  high-level convergence wrapper.
- `hemo1d.config`: declarative network configuration dataclasses and JSON
  loading/validation.
- `hemo1d.results`: result saving and probe plotting facade.

The public facade intentionally does not export low-level solver classes such
as `NetworkSolver`, CG/DG discretizations, or vessel factories.

## Canonical Subpackages

- `hemo1d.core`: shared physics, parameters, state containers, backend helpers,
  and Newton utilities.
- `hemo1d.topology`: endpoint and network graph structures.
- `hemo1d.boundary`: characteristic external boundary conditions, waveform
  helpers, and bifurcation/junction equations.
- `hemo1d.lumped`: lumped capillary-bed outlet data, equations, Newton solve,
  and coupling state.
- `hemo1d.solvers`: generic solver protocols, time configuration, generic
  vessel wrapper, network/model solver, and solver implementations.
- `hemo1d.solvers.cg`: Continuous Galerkin discretization, forms, mass solver,
  state, stepper, and factory.
- `hemo1d.solvers.dg`: Discontinuous Galerkin discretization, fluxes, limiter,
  residual, state, stepper, sampling, and factory.
- `hemo1d.observe`: probes, probe histories, diagnostics, and CG probe helpers.
- `hemo1d.io`: config readers, CSV scalar input readers, and output writers.
- `hemo1d.plotting`: optional Matplotlib plotting helpers.
- `hemo1d.convergence`: reusable convergence error, snapshot, plotting, and
  study helpers.

## Solver Flow

1. `HemodynamicModel` loads and stores declarative network config.
2. Boundary assignments, optional lumped beds, solver settings, and probes are
   attached to the model.
3. `build_vascular_network(...)` creates fresh mutable solver vessels and
   coupled outlet state for a run.
4. `NetworkSolver` advances vessels, junctions, external boundaries, and lumped
   beds without changing the mathematical kernels.
5. `Results` wraps the raw solver result for saving, plotting, capillary-bed
   histories, and metadata.

## Extension Points

- Add new boundary types under `hemo1d.boundary`.
- Add new outlet/coupling models under `hemo1d.lumped`.
- Add new solver methods under `hemo1d.solvers.<method>` and expose a factory.
- Keep shared formulas in `hemo1d.core.physics`; solver modules should not
  duplicate pressure laws, wave speeds, fluxes, or source terms.
- Add new observations under `hemo1d.observe`.

Numerical changes should be made deliberately and tested directly. Refactors
should preserve equations, signs, physical parameters, characteristic
conditions, boundary conditions, junction equations, and solver logic.
