from ag_survival_sim import Action, FarmState, ScenarioGenerator, StaticPolicy, TableCropModel, evaluate_policies
from ag_survival_sim.simulator import FarmSimulator


def build_simulator() -> FarmSimulator:
    return FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "low", "good", 180.0),
                ("corn", "low", "normal", 165.0),
                ("corn", "low", "drought", 105.0),
                ("soy", "medium", "good", 66.0),
                ("soy", "medium", "normal", 58.0),
                ("soy", "medium", "drought", 41.0),
            ]
        )
    )


def test_policy_evaluation_is_paired_and_returns_metrics() -> None:
    summary = evaluate_policies(
        simulator=build_simulator(),
        scenario_generator=ScenarioGenerator(seed=5),
        policies={
            "corn": StaticPolicy(Action("corn", "low")),
            "soy": StaticPolicy(Action("soy", "medium")),
        },
        initial_state=FarmState.initial(),
        horizon_years=8,
        num_paths=6,
    )

    assert set(summary.metrics) == {"corn", "soy"}
    assert summary.metrics["corn"].mean_survival_years >= 0.0
    assert summary.metrics["soy"].bankruptcy_rate >= 0.0
