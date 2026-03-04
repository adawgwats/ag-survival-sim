from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from statistics import mean

import numpy as np

from .crop_model import CropModel
from .finance import operating_cost, planned_operating_cost, realized_price
from .portfolio import (
    AllocationSlice,
    ChristensenKnightianPortfolioPolicy,
    GreedyMarginPortfolioPolicy,
    PortfolioAllocation,
    PortfolioPolicy,
    StaticPortfolioPolicy,
)
from .portfolio_simulator import PortfolioFarmSimulator, PortfolioStepRecord
from .scenario import AnnualScenario, ScenarioGenerator
from .types import Action, FarmState


_CANONICAL_REGIME_SCENARIOS = {
    "good": {"weather_yield_multiplier": 1.08, "market_price_multiplier": 0.95, "operating_cost_multiplier": 0.98, "basis_penalty": 0.02},
    "normal": {"weather_yield_multiplier": 1.00, "market_price_multiplier": 1.00, "operating_cost_multiplier": 1.00, "basis_penalty": 0.04},
    "drought": {"weather_yield_multiplier": 0.72, "market_price_multiplier": 1.15, "operating_cost_multiplier": 1.06, "basis_penalty": 0.08},
}


@dataclass(frozen=True)
class LearnedPortfolioConfig:
    horizon_years: int = 12
    training_paths: int = 12
    training_seed: int = 101
    random_exploration_policies: int = 4
    learning_rate: float = 0.03
    epochs: int = 240
    l2_penalty: float = 0.001
    candidate_random_samples: int = 18
    top_action_count: int = 4
    max_active_actions: int = 3
    max_share_per_action: float = 1.0
    max_share_per_crop: float = 1.0
    bankruptcy_penalty_per_acre: float = 15_000.0

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive.")
        if self.training_paths <= 0:
            raise ValueError("training_paths must be positive.")
        if self.random_exploration_policies < 0:
            raise ValueError("random_exploration_policies must be nonnegative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be nonnegative.")
        if self.candidate_random_samples < 0:
            raise ValueError("candidate_random_samples must be nonnegative.")
        if self.top_action_count <= 0:
            raise ValueError("top_action_count must be positive.")
        if self.max_active_actions <= 0:
            raise ValueError("max_active_actions must be positive.")
        if not 0.0 < self.max_share_per_action <= 1.0:
            raise ValueError("max_share_per_action must be in (0, 1].")
        if not 0.0 < self.max_share_per_crop <= 1.0:
            raise ValueError("max_share_per_crop must be in (0, 1].")
        if self.bankruptcy_penalty_per_acre < 0.0:
            raise ValueError("bankruptcy_penalty_per_acre must be nonnegative.")


@dataclass(frozen=True)
class LearnedPortfolioTrainingSummary:
    example_count: int
    target_mean: float
    target_std: float
    train_mse: float
    horizon_years: int
    training_paths: int
    training_seed: int


@dataclass(frozen=True)
class RandomPortfolioPolicy:
    candidate_generator: "PortfolioCandidateGenerator"
    seed: int

    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        rng = random.Random(hash((self.seed, state.year, round(state.cash, 2), round(state.debt, 2), scenario.year_index)))
        candidates = self.candidate_generator.generate(state, scenario, rng=rng, seed_allocations=())
        if not candidates:
            return PortfolioAllocation(())
        return candidates[rng.randrange(len(candidates))]


@dataclass(frozen=True)
class PortfolioCandidateGenerator:
    actions: tuple[Action, ...]
    crop_model: CropModel
    top_action_count: int = 4
    random_samples: int = 18
    max_active_actions: int = 3
    max_share_per_action: float = 1.0
    max_share_per_crop: float = 1.0

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if self.top_action_count <= 0:
            raise ValueError("top_action_count must be positive.")
        if self.random_samples < 0:
            raise ValueError("random_samples must be nonnegative.")
        if self.max_active_actions <= 0:
            raise ValueError("max_active_actions must be positive.")
        if not 0.0 < self.max_share_per_action <= 1.0:
            raise ValueError("max_share_per_action must be in (0, 1].")
        if not 0.0 < self.max_share_per_crop <= 1.0:
            raise ValueError("max_share_per_crop must be in (0, 1].")

    def generate(
        self,
        state: FarmState,
        scenario: AnnualScenario,
        *,
        rng: random.Random,
        seed_allocations: tuple[PortfolioAllocation, ...],
    ) -> list[PortfolioAllocation]:
        ranked_actions = self._rank_actions(state, scenario)
        top_actions = tuple(action for _score, action in ranked_actions[: self.top_action_count])
        candidates: list[PortfolioAllocation] = []

        for action in top_actions:
            allocation = self._allocation_from_weights(state, {action.key: 1.0})
            if allocation.total_acres > 0.0:
                candidates.append(allocation)

        for left_action, right_action in combinations(top_actions, 2):
            for left_share in (0.25, 0.5, 0.75):
                allocation = self._allocation_from_weights(
                    state,
                    {
                        left_action.key: left_share,
                        right_action.key: 1.0 - left_share,
                    },
                )
                if allocation.total_acres > 0.0:
                    candidates.append(allocation)

        if len(top_actions) >= 3:
            trio = top_actions[:3]
            for weights in (
                (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                (0.50, 0.25, 0.25),
                (0.25, 0.50, 0.25),
                (0.25, 0.25, 0.50),
            ):
                allocation = self._allocation_from_weights(
                    state,
                    {
                        trio[0].key: weights[0],
                        trio[1].key: weights[1],
                        trio[2].key: weights[2],
                    },
                )
                if allocation.total_acres > 0.0:
                    candidates.append(allocation)

        candidates.extend(allocation for allocation in seed_allocations if allocation.total_acres > 0.0)

        for _ in range(self.random_samples):
            active_count = rng.randint(1, min(self.max_active_actions, len(top_actions)))
            chosen_actions = list(rng.sample(list(top_actions), active_count))
            random_weights = [rng.random() for _ in chosen_actions]
            total = sum(random_weights)
            if total <= 0.0:
                continue
            normalized = [weight / total for weight in random_weights]
            allocation = self._allocation_from_weights(
                state,
                {
                    action.key: weight
                    for action, weight in zip(chosen_actions, normalized)
                },
            )
            if allocation.total_acres > 0.0:
                candidates.append(allocation)

        return self._unique_allocations(candidates)

    def _rank_actions(self, state: FarmState, scenario: AnnualScenario) -> list[tuple[float, Action]]:
        ranked: list[tuple[float, Action]] = []
        for action in self.actions:
            simulated_yield = self.crop_model.yield_per_acre(
                state=state,
                action=action,
                scenario=scenario,
            )
            margin = (
                simulated_yield * realized_price(action, scenario)
                - operating_cost(action, 1.0, scenario)
            )
            ranked.append((margin, action))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked

    def _allocation_from_weights(
        self,
        state: FarmState,
        weights_by_action_key: dict[tuple[str, str], float],
    ) -> PortfolioAllocation:
        available_capital = state.cash + state.remaining_credit
        max_acres_per_action = state.acres * self.max_share_per_action
        max_acres_per_crop = state.acres * self.max_share_per_crop

        crop_acres: dict[str, float] = {}
        slices: list[AllocationSlice] = []
        for action in self.actions:
            desired_weight = weights_by_action_key.get(action.key, 0.0)
            if desired_weight <= 0.0:
                continue
            desired_acres = desired_weight * state.acres
            crop_remaining = max_acres_per_crop - crop_acres.get(action.crop, 0.0)
            acres = min(desired_acres, max_acres_per_action, crop_remaining)
            if acres <= 1e-9:
                continue
            slices.append(AllocationSlice(action=action, acres=acres))
            crop_acres[action.crop] = crop_acres.get(action.crop, 0.0) + acres

        if not slices:
            return PortfolioAllocation(())

        total_planned_cost = sum(planned_operating_cost(s.action, s.acres) for s in slices)
        if total_planned_cost > available_capital > 0.0:
            scale = available_capital / total_planned_cost
            slices = [
                AllocationSlice(action=s.action, acres=s.acres * scale)
                for s in slices
                if s.acres * scale > 1e-6
            ]

        return PortfolioAllocation(tuple(slices))

    @staticmethod
    def _unique_allocations(candidates: list[PortfolioAllocation]) -> list[PortfolioAllocation]:
        unique: list[PortfolioAllocation] = []
        seen: set[tuple[tuple[tuple[str, str], int], ...]] = set()
        for candidate in candidates:
            key = tuple(
                sorted(
                    (
                        allocation_slice.action.key,
                        int(round(allocation_slice.acres * 10.0)),
                    )
                    for allocation_slice in candidate.nonzero_slices()
                )
            )
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique


@dataclass(frozen=True)
class LearnedRolloutPortfolioPolicy:
    actions: tuple[Action, ...]
    crop_model: CropModel
    horizon_years: int
    parameters: tuple[float, ...]
    target_mean: float
    target_std: float
    candidate_generator: PortfolioCandidateGenerator
    seed_policies: tuple[PortfolioPolicy, ...]
    training_summary: LearnedPortfolioTrainingSummary
    seed: int = 0

    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        seed_allocations = tuple(
            policy.choose_allocation(state, scenario)
            for policy in self.seed_policies
        )
        rng = random.Random(hash((self.seed, state.year, round(state.cash, 2), round(state.debt, 2), scenario.year_index)))
        candidates = self.candidate_generator.generate(
            state,
            scenario,
            rng=rng,
            seed_allocations=seed_allocations,
        )
        if not candidates:
            return PortfolioAllocation(())

        best_candidate = candidates[0]
        best_score = self._score(state, scenario, best_candidate)
        for candidate in candidates[1:]:
            score = self._score(state, scenario, candidate)
            if score > best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate

    def _score(
        self,
        state: FarmState,
        scenario: AnnualScenario,
        allocation: PortfolioAllocation,
    ) -> float:
        features = _featurize_state_allocation(
            state,
            scenario,
            allocation,
            self.actions,
            self.crop_model,
            self.horizon_years,
        )
        normalized_prediction = sum(parameter * feature for parameter, feature in zip(self.parameters, features))
        return normalized_prediction * self.target_std + self.target_mean


def train_learned_rollout_portfolio_policy(
    *,
    actions: tuple[Action, ...],
    crop_model: CropModel,
    initial_state: FarmState,
    config: LearnedPortfolioConfig,
    exploration_policies: dict[str, PortfolioPolicy],
) -> LearnedRolloutPortfolioPolicy:
    simulator = PortfolioFarmSimulator(crop_model=crop_model)
    scenario_generator = ScenarioGenerator(seed=config.training_seed)
    candidate_generator = PortfolioCandidateGenerator(
        actions=actions,
        crop_model=crop_model,
        top_action_count=config.top_action_count,
        random_samples=config.candidate_random_samples,
        max_active_actions=config.max_active_actions,
        max_share_per_action=config.max_share_per_action,
        max_share_per_crop=config.max_share_per_crop,
    )

    all_policies = dict(exploration_policies)
    for random_index in range(config.random_exploration_policies):
        all_policies[f"random_{random_index}"] = RandomPortfolioPolicy(
            candidate_generator=candidate_generator,
            seed=config.training_seed + random_index + 1,
        )

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    path_offset = 0
    for policy_name, policy in all_policies.items():
        del policy_name
        for local_path_index in range(config.training_paths):
            path = scenario_generator.generate_path(
                config.horizon_years,
                path_index=path_offset,
            )
            path_offset += 1
            state = initial_state
            steps: list[PortfolioStepRecord] = []
            states: list[FarmState] = []
            scenarios: list[AnnualScenario] = []
            allocations: list[PortfolioAllocation] = []
            for scenario in path:
                if not state.alive:
                    break
                allocation = policy.choose_allocation(state, scenario)
                record = simulator.step(state=state, allocation=allocation, scenario=scenario)
                steps.append(record)
                states.append(state)
                scenarios.append(scenario)
                allocations.append(allocation)
                state = record.ending_state
            if not steps:
                continue
            step_targets = _targets_from_steps(
                steps,
                acres=max(initial_state.acres, 1.0),
                bankruptcy_penalty_per_acre=config.bankruptcy_penalty_per_acre,
            )
            for state_row, scenario_row, allocation_row, target in zip(states, scenarios, allocations, step_targets):
                feature_rows.append(
                    _featurize_state_allocation(
                        state_row,
                        scenario_row,
                        allocation_row,
                        actions,
                        crop_model,
                        config.horizon_years,
                    )
                )
                targets.append(target)

    if not feature_rows:
        raise ValueError("learned policy training did not produce any examples.")

    target_mean = mean(targets)
    target_std = max((mean((target - target_mean) ** 2 for target in targets)) ** 0.5, 1e-6)
    normalized_targets = [(target - target_mean) / target_std for target in targets]
    parameters = _train_linear_model(
        feature_rows,
        normalized_targets,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2_penalty=config.l2_penalty,
    )
    predictions = [sum(parameter * feature for parameter, feature in zip(parameters, features)) for features in feature_rows]
    train_mse = mean((prediction - target) ** 2 for prediction, target in zip(predictions, normalized_targets))
    training_summary = LearnedPortfolioTrainingSummary(
        example_count=len(feature_rows),
        target_mean=target_mean,
        target_std=target_std,
        train_mse=train_mse,
        horizon_years=config.horizon_years,
        training_paths=config.training_paths,
        training_seed=config.training_seed,
    )
    return LearnedRolloutPortfolioPolicy(
        actions=actions,
        crop_model=crop_model,
        horizon_years=config.horizon_years,
        parameters=tuple(parameters),
        target_mean=target_mean,
        target_std=target_std,
        candidate_generator=candidate_generator,
        seed_policies=tuple(exploration_policies.values()),
        training_summary=training_summary,
        seed=config.training_seed,
    )


def _targets_from_steps(
    steps: list[PortfolioStepRecord],
    *,
    acres: float,
    bankruptcy_penalty_per_acre: float,
) -> list[float]:
    targets = [0.0 for _ in steps]
    cumulative_profit = 0.0
    eventual_bankruptcy = False
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        cumulative_profit += step.net_income / acres
        eventual_bankruptcy = eventual_bankruptcy or (not step.ending_state.alive)
        penalty = bankruptcy_penalty_per_acre if eventual_bankruptcy else 0.0
        targets[index] = cumulative_profit - penalty
    return targets


def _train_linear_model(
    feature_rows: list[list[float]],
    targets: list[float],
    *,
    learning_rate: float,
    epochs: int,
    l2_penalty: float,
) -> list[float]:
    del learning_rate, epochs
    design_matrix = np.asarray(feature_rows, dtype=float)
    target_vector = np.asarray(targets, dtype=float)
    feature_count = design_matrix.shape[1]
    if l2_penalty > 0.0:
        regularizer = np.sqrt(l2_penalty) * np.eye(feature_count, dtype=float)
        design_matrix = np.vstack((design_matrix, regularizer))
        target_vector = np.concatenate((target_vector, np.zeros(feature_count, dtype=float)))
    parameters, *_rest = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    return parameters.tolist()


def _featurize_state_allocation(
    state: FarmState,
    scenario: AnnualScenario,
    allocation: PortfolioAllocation,
    actions: tuple[Action, ...],
    crop_model: CropModel,
    horizon_years: int,
) -> list[float]:
    acres = max(state.acres, 1.0)
    deployable_capital = state.cash + state.remaining_credit
    planned_cost = sum(
        planned_operating_cost(allocation_slice.action, allocation_slice.acres)
        for allocation_slice in allocation.nonzero_slices()
    )
    planted_share = allocation.total_acres / acres

    shares_by_action = {action.key: 0.0 for action in actions}
    crop_shares = {"corn": 0.0, "soy": 0.0, "peanut": 0.0}
    irrigation_share = 0.0
    scenario_margin_per_acre = 0.0
    worst_margin_per_acre = 0.0

    for allocation_slice in allocation.nonzero_slices():
        share = allocation_slice.acres / acres
        shares_by_action[allocation_slice.action.key] = share
        crop_shares[allocation_slice.action.crop] = crop_shares.get(allocation_slice.action.crop, 0.0) + share
        if "irrigated" in allocation_slice.action.input_level:
            irrigation_share += share

        scenario_yield = crop_model.yield_per_acre(
            state=state,
            action=allocation_slice.action,
            scenario=scenario,
        )
        scenario_margin_per_acre += share * (
            scenario_yield * realized_price(allocation_slice.action, scenario)
            - operating_cost(allocation_slice.action, 1.0, scenario)
        )

        regime_margins: list[float] = []
        for regime in ("good", "normal", "drought"):
            regime_scenario = _scenario_with_regime(scenario, regime)
            regime_yield = crop_model.yield_per_acre(
                state=state,
                action=allocation_slice.action,
                scenario=regime_scenario,
            )
            regime_margins.append(
                regime_yield * realized_price(allocation_slice.action, regime_scenario)
                - operating_cost(allocation_slice.action, 1.0, regime_scenario)
            )
        worst_margin_per_acre += share * min(regime_margins)

    concentration = sum(share * share for share in shares_by_action.values())
    capital_utilization = planned_cost / max(deployable_capital, 1.0)

    features = [
        1.0,
        min(state.year / max(horizon_years, 1), 1.0),
        state.cash / acres / 1_000.0,
        state.debt / acres / 1_000.0,
        state.remaining_credit / acres / 1_000.0,
        state.land_mortgage_balance / acres / 1_000.0,
        state.cumulative_profit / acres / 1_000.0,
        state.consecutive_dscr_failures / 2.0,
        float(state.land_mortgage_grace_years_remaining == 0),
        capital_utilization,
        planted_share,
        concentration,
        irrigation_share,
        crop_shares.get("corn", 0.0),
        crop_shares.get("soy", 0.0),
        crop_shares.get("peanut", 0.0),
        scenario_margin_per_acre / 1_000.0,
        worst_margin_per_acre / 1_000.0,
        1.0 if scenario.weather_regime == "good" else 0.0,
        1.0 if scenario.weather_regime == "normal" else 0.0,
        1.0 if scenario.weather_regime == "drought" else 0.0,
    ]
    features.extend(shares_by_action[action.key] for action in actions)
    return features


def _scenario_with_regime(scenario: AnnualScenario, regime: str) -> AnnualScenario:
    params = _CANONICAL_REGIME_SCENARIOS[regime]
    return AnnualScenario(
        year_index=scenario.year_index,
        weather_regime=regime,
        weather_yield_multiplier=float(params["weather_yield_multiplier"]),
        market_price_multiplier=float(params["market_price_multiplier"]),
        operating_cost_multiplier=float(params["operating_cost_multiplier"]),
        basis_penalty=float(params["basis_penalty"]),
    )


def build_learning_exploration_policies(
    *,
    actions: tuple[Action, ...],
    crop_model: CropModel,
) -> dict[str, PortfolioPolicy]:
    corn_safe = _select_available_action(actions, "corn", ("irrigated_low", "medium", "low", "rainfed_low"))
    soy_medium = _select_available_action(actions, "soy", ("medium", "low"))
    peanut_medium = _select_available_action(actions, "peanut", ("medium", "low"))
    corn_cash_buffer = _select_available_action(actions, "corn", ("rainfed_low", "low", "irrigated_low", "medium"))
    peanut_low = _select_available_action(actions, "peanut", ("low", "medium"))

    equal_mix_shares = tuple(
        AllocationSlice(action=action, acres=share)
        for action, share in (
            (corn_safe, 1.0 / 3.0),
            (soy_medium, 1.0 / 3.0),
            (peanut_medium, 1.0 / 3.0),
        )
        if action is not None
    )
    cash_buffer_shares = tuple(
        AllocationSlice(action=action, acres=share)
        for action, share in (
            (corn_cash_buffer, 0.40),
            (soy_medium, 0.35),
            (peanut_low, 0.25),
        )
        if action is not None
    )

    return {
        "equal_mix": StaticPortfolioPolicy(
            shares=equal_mix_shares
        ),
        "cash_buffer_mix": StaticPortfolioPolicy(
            shares=cash_buffer_shares
        ),
        "greedy_margin": GreedyMarginPortfolioPolicy(
            actions=actions,
            crop_model=crop_model,
            max_share_per_action=0.6,
        ),
        "christensen_knightian": ChristensenKnightianPortfolioPolicy(
            actions=actions,
            crop_model=crop_model,
            max_share_per_action=0.5,
            max_share_per_crop=0.7,
        ),
    }


def _select_available_action(
    actions: tuple[Action, ...],
    crop: str,
    preferred_levels: tuple[str, ...],
) -> Action | None:
    actions_by_crop = [action for action in actions if action.crop == crop]
    if not actions_by_crop:
        return None
    for input_level in preferred_levels:
        for action in actions_by_crop:
            if action.input_level == input_level:
                return action
    return actions_by_crop[0]
