from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .crop_model import CropModel
from .finance import operating_cost, planned_operating_cost, realized_price
from .scenario import AnnualScenario
from .types import Action, FarmState


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
