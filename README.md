# Ag Survival Sim

`ag-survival-sim` is an agricultural planning benchmark for comparing policies under shared weather, price, and finance scenarios.

The benchmark is designed to answer:

`If two policies face the same plausible farm futures, which one survives longer and fails less often?`

## Current scope

- fixed-acreage farm simulation
- discrete crop and input actions
- paired evaluation under identical scenario draws
- finance and bankruptcy logic
- selective-observation hooks for later minimax training benchmarks
- full DSSAT executable integration
- DSSAT-style table model for tests and demos

## DSSAT position

The intended crop engine is **full DSSAT**, not a synthetic yield model.

The package now supports:

- calling an installed DSSAT executable in batch mode
- preparing per-run workspaces from a template directory
- parsing `Summary.OUT`
- selecting the correct DSSAT summary row for the current run

The table model still exists, but only as:

- a lightweight demo backend
- a test fixture
- a fallback when DSSAT is not installed locally

## Full DSSAT quickstart

```python
from pathlib import Path

from ag_survival_sim import (
    Action,
    DSSATExecutableConfig,
    DSSATExecutableCropModel,
    FarmState,
    ScenarioGenerator,
    TemplateDSSATRunFactory,
)


def render_scenario(working_dir, state, action, scenario):
    # Update DSSAT experiment, weather, or management files in working_dir here.
    # This hook is where your simulator-specific DSSAT inputs get written.
    return None


crop_model = DSSATExecutableCropModel(
    run_factory=TemplateDSSATRunFactory(
        template_dir=Path("templates/corn-soy-base"),
        workspace_root=Path("runs"),
        batch_file="DSSBatch.v48",
        scenario_renderer=render_scenario,
        selector={"TRNO": 1},
    ),
    config=DSSATExecutableConfig(
        executable=(r"C:\DSSAT48\DSCSM048.EXE",),
    ),
)
```

Notes:

- The package does **not** ship DSSAT.
- You need a local DSSAT installation and valid experiment inputs.
- `Summary.OUT` parsing is built in.
- Unit alignment is your responsibility. If DSSAT outputs `HWAM` in raw DSSAT units, pass a `yield_transform` in `DSSATExecutableConfig` so the simulator and finance model use consistent units.

## Quickstart

```python
from ag_survival_sim import (
    Action,
    FarmPolicy,
    FarmSimulator,
    FarmState,
    ScenarioGenerator,
    StaticPolicy,
    TableCropModel,
    evaluate_policies,
)

crop_model = TableCropModel.from_records(
    [
        ("corn", "low", "normal", 165.0),
        ("corn", "low", "drought", 105.0),
        ("soy", "medium", "normal", 58.0),
        ("soy", "medium", "drought", 41.0),
    ]
)

simulator = FarmSimulator(crop_model=crop_model)
generator = ScenarioGenerator(seed=7)

baseline = StaticPolicy(Action(crop="corn", input_level="low"))
alternative = StaticPolicy(Action(crop="soy", input_level="medium"))

summary = evaluate_policies(
    simulator=simulator,
    scenario_generator=generator,
    policies={"baseline": baseline, "alternative": alternative},
    initial_state=FarmState.initial(),
    horizon_years=10,
    num_paths=50,
)

print(summary.metrics["baseline"].mean_survival_years)
```

That example uses the demo table model so the package remains runnable without DSSAT.

## Metrics

Primary metrics:

- mean survival years
- median survival years
- bankruptcy rate
- fifth-percentile terminal wealth

Secondary metrics:

- mean terminal wealth
- mean cumulative profit

## Observation process

The simulator produces full latent outcomes and can also apply a selective-observation rule. That keeps the benchmark compatible with training pipelines that need both:

- full outcomes for evaluation
- selectively observed outcomes for training data generation
