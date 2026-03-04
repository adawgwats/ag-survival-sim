from pathlib import Path

from ag_survival_sim import Action, FarmState, ScenarioGenerator, StaticPolicy, TableCropModel, evaluate_policies
from ag_survival_sim.simulator import FarmSimulator
from ag_survival_sim.visualization import plot_policy_action_traces, plot_policy_profit_traces


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


def test_plot_policy_action_traces_writes_png(tmp_path: Path) -> None:
    summary = evaluate_policies(
        simulator=build_simulator(),
        scenario_generator=ScenarioGenerator(seed=5),
        policies={
            "corn": StaticPolicy(Action("corn", "low")),
            "soy": StaticPolicy(Action("soy", "medium")),
        },
        initial_state=FarmState.initial(),
        horizon_years=4,
        num_paths=2,
    )

    output = plot_policy_action_traces(
        policy_evaluations=summary.evaluations,
        path_index=0,
        output_path=tmp_path / "policy_trace.png",
        title="paired policy trace",
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_policy_profit_traces_writes_png(tmp_path: Path) -> None:
    summary = evaluate_policies(
        simulator=build_simulator(),
        scenario_generator=ScenarioGenerator(seed=5),
        policies={
            "corn": StaticPolicy(Action("corn", "low")),
            "soy": StaticPolicy(Action("soy", "medium")),
        },
        initial_state=FarmState.initial(),
        horizon_years=4,
        num_paths=2,
    )

    output = plot_policy_profit_traces(
        policy_evaluations=summary.evaluations,
        path_index=0,
        output_path=tmp_path / "profit_trace.png",
        title="paired profit trace",
    )

    assert output.exists()
    assert output.stat().st_size > 0
