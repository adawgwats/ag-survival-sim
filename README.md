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
- DSSAT crop discovery across the installed DSSAT root
- benchmark-ready DSSAT bundles for maize, soybean, wheat, rice, peanut, and sunflower
- DSSAT-style table model for tests and demos

## DSSAT position

The intended crop engine is **full DSSAT**, not a synthetic yield model.

The package now supports:

- calling an installed DSSAT executable in batch mode
- preparing per-run workspaces from installed DSSAT experiment bundles
- parsing `Summary.OUT`
- selecting the correct DSSAT summary row for the current run
- running official DSSAT example suites from an installed DSSAT root

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
- By default, helper utilities look for DSSAT at `DSSAT_ROOT` or `C:\DSSAT48`.

## Official DSSAT example sweep

To verify a real DSSAT installation and collect benchmark-ready yield data from official examples:

```bash
ag-survival-dssat-suite --crop-directory Maize --archive-root dssat_runs/maize
```

That command:

- discovers DSSAT at `DSSAT_ROOT` or `C:\DSSAT48`
- runs the official maize `*.MZX` examples with the real `DSCSM048.EXE`
- parses each generated `Summary.OUT`
- prints a compact table with per-experiment treatment counts and `HWAM` ranges
- optionally archives the raw DSSAT outputs for later inspection

Python API:

```python
from ag_survival_sim import format_dssat_example_results, run_dssat_example_suite

results = run_dssat_example_suite(crop_directory="Maize", archive_root="dssat_runs/maize")
print(format_dssat_example_results(results))
```

To discover which crop directories and experiment templates exist in a local DSSAT install:

```python
from ag_survival_sim import discover_dssat_crop_inventory, format_crop_inventory

inventory = discover_dssat_crop_inventory()
print(format_crop_inventory(inventory))
```

To collect a treatment-level CSV across all discovered DSSAT crop directories:

```bash
ag-survival-dssat-all-crops --output-csv dssat_runs/all_crops/dssat_treatment_rows.csv
```

That sweep:

- runs official DSSAT example experiments across every discovered crop directory
- exports one CSV row per `Summary.OUT` treatment row
- skips DSSAT workflow directories such as `ClimateChange`, `Sequence`, `Spatial`, `Seasonal`, and `YieldForecast`

The installed DSSAT root may contain many crop directories. The simulation package can discover all of them, but the current benchmark bundles are intentionally narrower:

- `iowa_maize`
- `georgia_soybean`
- `kansas_wheat`
- `dtsp_rice`
- `georgia_peanut`
- `uafd_sunflower`

## DSSAT-backed benchmark bundles

The package now includes benchmark-ready DSSAT simulator paths built from official experiment files:

```bash
ag-survival-dssat-benchmark --benchmark iowa_maize --paths 8 --horizon 6
```

Available benchmark bundles:

- `iowa_maize` from `IUAF9901.MZX`
- `georgia_soybean` from `UFGA8401.SBX`
- `kansas_wheat` from `KSAS8101.WHX`
- `dtsp_rice` from `DTSP8502.RIX`
- `georgia_peanut` from `UFGA8701.PNX`
- `uafd_sunflower` from `UAFD0801.SUX`

These paths currently:

- uses the real installed DSSAT executable
- map low/medium action levels to official DSSAT treatment numbers
- writes per-run experiment directories instead of mutating the shared DSSAT install
- perturbs the weather file to create `good`, `normal`, and `drought` years
- converts DSSAT `HWAM` output from `kg/ha` into crop-specific economic units for the finance model
  - bushels/acre for maize, soybean, and wheat
  - cwt/acre for rice and sunflower
  - short tons/acre for peanut

Python API:

```python
from ag_survival_sim import build_benchmark_simulator

simulator = build_benchmark_simulator(
    "georgia_soybean",
    workspace_root="dssat_runs/georgia_soybean",
)
```

This is still a narrow bridge into the full benchmark:

- only a small benchmark-ready subset of installed DSSAT crops is economically configured
- two action levels per benchmark crop
- stylized weather perturbations rather than a calibrated weather generator
- rice currently has a weaker weather-response signal than peanut and sunflower under the default perturbations

But the yield path is now generated by real DSSAT runs, not a table lookup.

## Training dataset export

The benchmark can now export per-year training examples that include both:

- latent full outcomes from the simulator
- selectively observed labels produced by the observation process

Example:

```python
from ag_survival_sim import (
    Action,
    FarmState,
    ObservationProcess,
    ScenarioGenerator,
    SelectiveObservationRule,
    StaticPolicy,
    build_iowa_maize_simulator,
    generate_training_examples,
)

examples = generate_training_examples(
    simulator=build_iowa_maize_simulator(workspace_root="dssat_runs/iowa_maize_dataset"),
    scenario_generator=ScenarioGenerator(seed=13),
    policy=StaticPolicy(Action("corn", "medium")),
    observation_process=ObservationProcess(SelectiveObservationRule(seed=7)),
    initial_state=FarmState.initial(cash=300000, debt=100000, credit_limit=175000),
    horizon_years=4,
    num_paths=6,
)
```

Each exported example includes:

- pre-decision farm state
- chosen action
- weather regime
- latent yield / price / net income
- observed or missing labels
- a simple default `group_id` for robust-training experiments

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
