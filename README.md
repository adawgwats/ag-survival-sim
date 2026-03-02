# Ag Survival Sim

`ag-survival-sim` is a small agricultural planning benchmark for comparing policies under shared weather, price, and finance scenarios.

The benchmark is designed to answer:

`If two policies face the same plausible farm futures, which one survives longer and fails less often?`

## Current scope

- fixed-acreage farm simulation
- discrete crop and input actions
- paired evaluation under identical scenario draws
- finance and bankruptcy logic
- selective-observation hooks for later minimax training benchmarks
- DSSAT-style crop model interface

## DSSAT position

The package is structured so crop yields come from a `CropModel` interface. `v0` includes a tabular DSSAT-style adapter and tests use in-memory tables.

That means:

- the simulator is runnable today
- real DSSAT outputs can be plugged in later through the same interface
- the benchmark does not pretend to reproduce full DSSAT internals

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
