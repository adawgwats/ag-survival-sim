from __future__ import annotations

import argparse

from .crop_model import TableCropModel
from .dssat_suite import format_dssat_example_results, run_dssat_example_suite
from .evaluation import evaluate_policies
from .policy import GreedyProfitPolicy, StaticPolicy
from .scenario import ScenarioGenerator
from .simulator import FarmSimulator
from .types import Action, FarmState


def build_demo_simulator() -> FarmSimulator:
    crop_model = TableCropModel.from_records(
        [
            ("corn", "low", "good", 180.0),
            ("corn", "low", "normal", 165.0),
            ("corn", "low", "drought", 105.0),
            ("corn", "medium", "good", 205.0),
            ("corn", "medium", "normal", 185.0),
            ("corn", "medium", "drought", 115.0),
            ("soy", "low", "good", 61.0),
            ("soy", "low", "normal", 54.0),
            ("soy", "low", "drought", 38.0),
            ("soy", "medium", "good", 66.0),
            ("soy", "medium", "normal", 58.0),
            ("soy", "medium", "drought", 41.0),
            ("wheat", "low", "good", 74.0),
            ("wheat", "low", "normal", 69.0),
            ("wheat", "low", "drought", 46.0),
        ]
    )
    return FarmSimulator(crop_model=crop_model)


def run_demo() -> None:
    simulator = build_demo_simulator()
    generator = ScenarioGenerator(seed=13)

    summary = evaluate_policies(
        simulator=simulator,
        scenario_generator=generator,
        policies={
            "corn_low": StaticPolicy(Action("corn", "low")),
            "greedy": GreedyProfitPolicy(
                actions=(
                    Action("corn", "low"),
                    Action("corn", "medium"),
                    Action("soy", "medium"),
                    Action("wheat", "low"),
                )
            ),
        },
        initial_state=FarmState.initial(),
        horizon_years=12,
        num_paths=40,
    )

    for policy_name, metrics in summary.metrics.items():
        print(policy_name)
        print(f"  mean survival years: {metrics.mean_survival_years:.2f}")
        print(f"  bankruptcy rate: {metrics.bankruptcy_rate:.2%}")
        print(f"  mean terminal wealth: {metrics.mean_terminal_wealth:,.0f}")
        print(f"  5th pct terminal wealth: {metrics.fifth_percentile_terminal_wealth:,.0f}")


def run_dssat_suite_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run official DSSAT example experiments and summarize Summary.OUT yields."
    )
    parser.add_argument("--root", type=str, default=None, help="DSSAT install root. Defaults to DSSAT_ROOT or C:\\DSSAT48.")
    parser.add_argument("--crop-directory", type=str, default="Maize", help="DSSAT crop directory to run.")
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        default=None,
        help="Specific DSSAT experiment file to run. Repeat to run multiple examples.",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Optional directory where each experiment's DSSAT output files are copied.",
    )
    args = parser.parse_args()

    results = run_dssat_example_suite(
        root=args.root,
        crop_directory=args.crop_directory,
        experiments=args.experiments,
        archive_root=args.archive_root,
    )
    print(format_dssat_example_results(results))


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
