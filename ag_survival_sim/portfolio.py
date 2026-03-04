from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .crop_model import CropModel
from .finance import operating_cost, planned_operating_cost, realized_price
from .scenario import AnnualScenario
from .types import Action, FarmState


_CANONICAL_REGIME_SCENARIOS = {
    "good": {"weather_yield_multiplier": 1.08, "market_price_multiplier": 0.95, "operating_cost_multiplier": 0.98, "basis_penalty": 0.02},
    "normal": {"weather_yield_multiplier": 1.00, "market_price_multiplier": 1.00, "operating_cost_multiplier": 1.00, "basis_penalty": 0.04},
    "drought": {"weather_yield_multiplier": 0.72, "market_price_multiplier": 1.15, "operating_cost_multiplier": 1.06, "basis_penalty": 0.08},
}


@dataclass(frozen=True)
class AllocationSlice:
    action: Action
    acres: float

    def __post_init__(self) -> None:
        if self.acres < 0.0:
            raise ValueError("allocation acres must be nonnegative.")


@dataclass(frozen=True)
class PortfolioAllocation:
    slices: tuple[AllocationSlice, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[str, str]] = set()
        for allocation_slice in self.slices:
            if allocation_slice.action.key in seen:
                raise ValueError("portfolio allocation cannot contain duplicate actions.")
            seen.add(allocation_slice.action.key)

    @property
    def total_acres(self) -> float:
        return sum(allocation_slice.acres for allocation_slice in self.slices)

    def nonzero_slices(self) -> tuple[AllocationSlice, ...]:
        return tuple(allocation_slice for allocation_slice in self.slices if allocation_slice.acres > 0.0)


class PortfolioPolicy(Protocol):
    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        ...


@dataclass(frozen=True)
class StaticPortfolioPolicy:
    shares: tuple[AllocationSlice, ...]

    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        del scenario
        share_total = sum(allocation_slice.acres for allocation_slice in self.shares)
        if share_total <= 0.0:
            return PortfolioAllocation(())
        if share_total > 1.0 + 1e-9:
            raise ValueError("static portfolio shares must sum to at most 1.0.")

        available_capital = state.cash + state.remaining_credit
        desired_slices = [
            AllocationSlice(
                action=allocation_slice.action,
                acres=allocation_slice.acres * state.acres,
            )
            for allocation_slice in self.shares
            if allocation_slice.acres > 0.0
        ]
        planned_cost = sum(planned_operating_cost(s.action, s.acres) for s in desired_slices)
        if planned_cost <= 0.0 or planned_cost <= available_capital:
            return PortfolioAllocation(tuple(desired_slices))

        scale = available_capital / planned_cost if planned_cost > 0.0 else 0.0
        return PortfolioAllocation(
            tuple(
                AllocationSlice(action=s.action, acres=s.acres * scale)
                for s in desired_slices
                if s.acres * scale > 1e-9
            )
        )


@dataclass(frozen=True)
class GreedyMarginPortfolioPolicy:
    actions: tuple[Action, ...]
    crop_model: CropModel
    max_share_per_action: float = 1.0

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not 0.0 < self.max_share_per_action <= 1.0:
            raise ValueError("max_share_per_action must be in (0, 1].")

    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        per_acre_scores: list[tuple[float, Action]] = []
        for action in self.actions:
            simulated_yield = self.crop_model.yield_per_acre(
                state=state,
                action=action,
                scenario=scenario,
            )
            score = (
                simulated_yield * realized_price(action, scenario)
                - operating_cost(action, 1.0, scenario)
            )
            per_acre_scores.append((score, action))
        per_acre_scores.sort(reverse=True, key=lambda item: item[0])

        remaining_acres = state.acres
        remaining_capital = state.cash + state.remaining_credit
        max_acres_per_action = state.acres * self.max_share_per_action
        selected: list[AllocationSlice] = []
        for score, action in per_acre_scores:
            if remaining_acres <= 1e-9 or remaining_capital <= 1e-9:
                break
            cost_per_acre = planned_operating_cost(action, 1.0)
            if cost_per_acre <= 0.0:
                continue
            affordable_acres = remaining_capital / cost_per_acre
            if score <= 0.0 and selected:
                break
            acres = min(remaining_acres, affordable_acres, max_acres_per_action)
            if acres <= 1e-9:
                continue
            selected.append(AllocationSlice(action=action, acres=acres))
            remaining_acres -= acres
            remaining_capital -= cost_per_acre * acres

        return PortfolioAllocation(tuple(selected))


@dataclass(frozen=True)
class ChristensenKnightianPortfolioPolicy:
    actions: tuple[Action, ...]
    crop_model: CropModel
    max_share_per_action: float = 0.5
    max_share_per_crop: float = 0.7
    base_ambiguity: float = 0.35
    max_ambiguity: float = 0.90

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty.")
        if not 0.0 < self.max_share_per_action <= 1.0:
            raise ValueError("max_share_per_action must be in (0, 1].")
        if not 0.0 < self.max_share_per_crop <= 1.0:
            raise ValueError("max_share_per_crop must be in (0, 1].")
        if not 0.0 <= self.base_ambiguity <= self.max_ambiguity <= 1.0:
            raise ValueError("ambiguity parameters must satisfy 0 <= base <= max <= 1.")

    def choose_allocation(self, state: FarmState, scenario: AnnualScenario) -> PortfolioAllocation:
        ambiguity = self._ambiguity_level(state)
        remaining_acres = state.acres
        remaining_capital = state.cash + state.remaining_credit
        max_acres_per_action = state.acres * self.max_share_per_action
        max_acres_per_crop = state.acres * self.max_share_per_crop
        crop_acres: dict[str, float] = {}

        scored_actions = sorted(
            ((self._robust_margin(action, state, scenario, ambiguity), action) for action in self.actions),
            reverse=True,
            key=lambda item: item[0],
        )

        selected: list[AllocationSlice] = []
        for robust_score, action in scored_actions:
            if remaining_acres <= 1e-9 or remaining_capital <= 1e-9:
                break
            if robust_score <= 0.0 and selected:
                break
            cost_per_acre = planned_operating_cost(action, 1.0)
            if cost_per_acre <= 0.0:
                continue
            affordable_acres = remaining_capital / cost_per_acre
            crop_remaining = max_acres_per_crop - crop_acres.get(action.crop, 0.0)
            acres = min(remaining_acres, affordable_acres, max_acres_per_action, crop_remaining)
            if acres <= 1e-9:
                continue
            selected.append(AllocationSlice(action=action, acres=acres))
            remaining_acres -= acres
            remaining_capital -= cost_per_acre * acres
            crop_acres[action.crop] = crop_acres.get(action.crop, 0.0) + acres

        return PortfolioAllocation(tuple(selected))

    def _ambiguity_level(self, state: FarmState) -> float:
        deployable_cash_per_acre = (state.cash + state.remaining_credit) / max(state.acres, 1.0)
        liquidity_stress = max(0.0, (900.0 - deployable_cash_per_acre) / 900.0)
        profit_stress = max(0.0, -state.cumulative_profit / max(state.land_mortgage_balance, 1.0))
        mortgage_stress = 0.25 if state.land_mortgage_grace_years_remaining == 0 else 0.0
        ambiguity = self.base_ambiguity + 0.30 * liquidity_stress + 0.30 * min(profit_stress, 1.0) + mortgage_stress
        return min(max(ambiguity, self.base_ambiguity), self.max_ambiguity)

    def _robust_margin(
        self,
        action: Action,
        state: FarmState,
        scenario: AnnualScenario,
        ambiguity: float,
    ) -> float:
        regime_margins: list[float] = []
        for regime in ("good", "normal", "drought"):
            regime_scenario = self._scenario_with_regime(scenario, regime)
            simulated_yield = self.crop_model.yield_per_acre(
                state=state,
                action=action,
                scenario=regime_scenario,
            )
            regime_margin = (
                simulated_yield * realized_price(action, regime_scenario)
                - operating_cost(action, 1.0, regime_scenario)
            )
            regime_margins.append(regime_margin)
        average_margin = sum(regime_margins) / len(regime_margins)
        worst_margin = min(regime_margins)
        return (1.0 - ambiguity) * average_margin + ambiguity * worst_margin

    @staticmethod
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
