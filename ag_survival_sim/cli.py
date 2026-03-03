from __future__ import annotations

import argparse

from .crop_model import TableCropModel
from .dssat_benchmarks import (
    BENCHMARK_DEFINITIONS,
    build_benchmark_simulator,
    discover_dssat_crop_inventory,
    format_crop_inventory,
    get_benchmark_definition,
)
from .dssat_suite import (
    format_dssat_all_crops_summary,
    format_dssat_example_results,
    run_dssat_all_crops_sweep,
    run_dssat_example_suite,
)
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


def run_iowa_maize_demo_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run a DSSAT-backed survival benchmark."
    )
    parser.add_argument("--root", type=str, default=None, help="DSSAT install root. Defaults to DSSAT_ROOT or C:\\DSSAT48.")
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_DEFINITIONS),
        default="iowa_maize",
        help="Benchmark-ready DSSAT crop bundle to run.",
    )
    parser.add_argument("--workspace-root", type=str, default="dssat_runs/dssat_benchmark_demo", help="Directory for per-run DSSAT workspaces.")
    parser.add_argument("--seed", type=int, default=13, help="Scenario RNG seed.")
    parser.add_argument("--paths", type=int, default=8, help="Number of paired scenario paths.")
    parser.add_argument("--horizon", type=int, default=6, help="Simulation horizon in years.")
    parser.add_argument("--cash", type=float, default=300000.0, help="Initial farm cash.")
    parser.add_argument("--debt", type=float, default=100000.0, help="Initial farm debt.")
    parser.add_argument("--credit-limit", type=float, default=175000.0, help="Initial credit limit.")
    parser.add_argument("--acres", type=float, default=500.0, help="Farm acreage.")
    args = parser.parse_args()

    benchmark = get_benchmark_definition(args.benchmark)
    simulator = build_benchmark_simulator(
        args.benchmark,
        dssat_root=args.root,
        workspace_root=args.workspace_root,
    )
    policies = {
        f"{action.crop}_{action.input_level}": StaticPolicy(action)
        for action in benchmark.actions
    }
    summary = evaluate_policies(
        simulator=simulator,
        scenario_generator=ScenarioGenerator(seed=args.seed),
        policies=policies,
        initial_state=FarmState.initial(
            cash=args.cash,
            debt=args.debt,
            credit_limit=args.credit_limit,
            acres=args.acres,
        ),
        horizon_years=args.horizon,
        num_paths=args.paths,
    )

    for policy_name, metrics in summary.metrics.items():
        print(policy_name)
        print(f"  mean survival years: {metrics.mean_survival_years:.2f}")
        print(f"  bankruptcy rate: {metrics.bankruptcy_rate:.2%}")
        print(f"  mean terminal wealth: {metrics.mean_terminal_wealth:,.0f}")
        print(f"  5th pct terminal wealth: {metrics.fifth_percentile_terminal_wealth:,.0f}")
        print(f"  mean cumulative profit: {metrics.mean_cumulative_profit:,.0f}")


def run_dssat_catalog_cli() -> None:
    parser = argparse.ArgumentParser(
        description="List discovered DSSAT crops and benchmark-ready crop bundles."
    )
    parser.add_argument("--root", type=str, default=None, help="DSSAT install root. Defaults to DSSAT_ROOT or C:\\DSSAT48.")
    args = parser.parse_args()

    inventory = discover_dssat_crop_inventory(dssat_root=args.root)
    print("Discovered DSSAT crops with experiment templates")
    print(format_crop_inventory(inventory))
    print()
    print("Benchmark-ready crop bundles")
    for name in sorted(BENCHMARK_DEFINITIONS):
        definition = BENCHMARK_DEFINITIONS[name]
        action_labels = ", ".join(f"{action.crop}/{action.input_level}" for action in definition.actions)
        print(f"{name}: {definition.crop_directory}/{definition.experiment_file} -> {action_labels}")


def run_dssat_all_crops_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run official DSSAT examples across all discovered crop directories and export treatment rows."
    )
    parser.add_argument("--root", type=str, default=None, help="DSSAT install root. Defaults to DSSAT_ROOT or C:\\DSSAT48.")
    parser.add_argument(
        "--crop-directory",
        dest="crop_directories",
        action="append",
        default=None,
        help="Optional crop directory to include. Repeat to limit the sweep.",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Optional directory where raw DSSAT outputs are archived by crop and experiment.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="dssat_runs/all_crops/dssat_treatment_rows.csv",
        help="CSV file where treatment-level DSSAT rows are exported.",
    )
    args = parser.parse_args()

    _results, summary = run_dssat_all_crops_sweep(
        root=args.root,
        crop_directories=args.crop_directories,
        archive_root=args.archive_root,
        output_csv=args.output_csv,
    )
    print(format_dssat_all_crops_summary(summary))
    print()
    print(f"treatment rows exported to: {args.output_csv}")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
