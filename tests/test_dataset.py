from __future__ import annotations

from ag_survival_sim import (
    Action,
    FarmState,
    ObservationProcess,
    ScenarioGenerator,
    SelectiveObservationRule,
    StaticPolicy,
    TableCropModel,
    TrainingExample,
    default_group_id,
    generate_training_examples,
)
from ag_survival_sim.simulator import FarmSimulator


def test_generate_training_examples_produces_latent_and_observed_fields() -> None:
    simulator = FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "low", "good", 180.0),
                ("corn", "low", "normal", 160.0),
                ("corn", "low", "drought", 90.0),
            ]
        )
    )
    examples = generate_training_examples(
        simulator=simulator,
        scenario_generator=ScenarioGenerator(seed=4),
        policy=StaticPolicy(Action("corn", "low")),
        observation_process=ObservationProcess(
            SelectiveObservationRule(seed=7, distressed_penalty=0.8)
        ),
        initial_state=FarmState.initial(),
        horizon_years=4,
        num_paths=3,
    )

    assert examples
    assert all(example.group_id in {"stable", "distressed"} for example in examples)
    assert all(example.latent_yield_per_acre > 0 for example in examples)
    assert any(example.label_observed is False for example in examples)
    assert any(example.group_id == "distressed" for example in examples)


def test_default_group_id_marks_drought_and_losses_as_distressed() -> None:
    example = TrainingExample(
        path_index=0,
        step_index=0,
        year=0,
        crop="corn",
        input_level="low",
        weather_regime="drought",
        cash=50_000.0,
        debt=150_000.0,
        credit_limit=60_000.0,
        acres=500.0,
        land_value_per_acre=4_000.0,
        land_mortgage_balance=2_000_000.0,
        land_mortgage_rate=0.045,
        land_mortgage_years_remaining=30,
        land_mortgage_grace_years_remaining=2,
        latent_yield_per_acre=50.0,
        latent_net_income=-12_000.0,
        latent_price=4.5,
        observed_yield_per_acre=None,
        observed_net_income=None,
        observed_price=None,
        label_observed=False,
        group_id="",
        farm_alive_next_year=False,
    )

    assert default_group_id(example) == "distressed"
