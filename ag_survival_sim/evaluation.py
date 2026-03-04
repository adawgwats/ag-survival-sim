from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from .policy import FarmPolicy
from .scenario import ScenarioGenerator
from .simulator import FarmSimulator
from .types import FarmState, FarmStepRecord


@dataclass(frozen=True)
class PathResult:
    steps: list[FarmStepRecord]

    @property
    def final_state(self) -> FarmState:
        return self.steps[-1].ending_state if self.steps else FarmState.initial()

    @property
    def survival_years(self) -> int:
        if not self.steps:
            return 0
        for index, step in enumerate(self.steps, start=1):
            if not step.ending_state.alive:
                return index
        return len(self.steps)

    @property
    def bankrupt(self) -> bool:
        return bool(self.steps) and not self.steps[-1].ending_state.alive


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    path_results: list[PathResult]


@dataclass(frozen=True)
class PolicyMetrics:
    mean_survival_years: float
    median_survival_years: float
    full_horizon_survival_rate: float
    bankruptcy_rate: float
    mean_terminal_wealth: float
    fifth_percentile_terminal_wealth: float
    mean_cumulative_profit: float


@dataclass(frozen=True)
class ScenarioEvaluationSummary:
    evaluations: dict[str, PolicyEvaluation]
    metrics: dict[str, PolicyMetrics]


def evaluate_policies(
    *,
    simulator: FarmSimulator,
    scenario_generator: ScenarioGenerator,
    policies: dict[str, FarmPolicy],
    initial_state: FarmState,
    horizon_years: int,
    num_paths: int,
) -> ScenarioEvaluationSummary:
    evaluations: dict[str, PolicyEvaluation] = {}
    metrics: dict[str, PolicyMetrics] = {}

    for policy_name, policy in policies.items():
        path_results: list[PathResult] = []
        for path_index in range(num_paths):
            path = scenario_generator.generate_path(horizon_years, path_index=path_index)
            state = initial_state
            steps: list[FarmStepRecord] = []
            for scenario in path:
                if not state.alive:
                    break
                action = policy.choose_action(state, scenario)
                record = simulator.step(state=state, action=action, scenario=scenario)
                steps.append(record)
                state = record.ending_state
            path_results.append(PathResult(steps=steps))

        evaluations[policy_name] = PolicyEvaluation(policy_name=policy_name, path_results=path_results)
        metrics[policy_name] = summarize_policy(path_results)

    return ScenarioEvaluationSummary(evaluations=evaluations, metrics=metrics)


def summarize_policy(path_results: list[PathResult]) -> PolicyMetrics:
    if not path_results:
        raise ValueError("path_results must not be empty.")

    survival_years = [result.survival_years for result in path_results]
    terminal_wealth = [
        result.final_state.cash - result.final_state.debt
        for result in path_results
    ]
    cumulative_profit = [
        result.final_state.cumulative_profit
        for result in path_results
    ]
    bankruptcies = sum(1 for result in path_results if result.bankrupt)
    full_horizon_survivals = sum(
        1
        for result in path_results
        if result.steps and result.steps[-1].ending_state.alive
    )
    ordered_terminal_wealth = sorted(terminal_wealth)
    fifth_index = max(int(0.05 * len(ordered_terminal_wealth)) - 1, 0)

    return PolicyMetrics(
        mean_survival_years=mean(survival_years),
        median_survival_years=median(survival_years),
        full_horizon_survival_rate=full_horizon_survivals / len(path_results),
        bankruptcy_rate=bankruptcies / len(path_results),
        mean_terminal_wealth=mean(terminal_wealth),
        fifth_percentile_terminal_wealth=ordered_terminal_wealth[fifth_index],
        mean_cumulative_profit=mean(cumulative_profit),
    )
