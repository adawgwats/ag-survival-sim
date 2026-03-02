from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .finance import ECONOMICS_BY_ACTION, realized_price
from .scenario import AnnualScenario
from .types import Action, FarmState


class FarmPolicy(Protocol):
    def choose_action(self, state: FarmState, scenario: AnnualScenario) -> Action:
        ...


@dataclass(frozen=True)
class StaticPolicy:
    action: Action

    def choose_action(self, state: FarmState, scenario: AnnualScenario) -> Action:
        del state, scenario
        return self.action


@dataclass(frozen=True)
class GreedyProfitPolicy:
    actions: tuple[Action, ...]

    def choose_action(self, state: FarmState, scenario: AnnualScenario) -> Action:
        best_action = self.actions[0]
        best_score = float("-inf")

        for action in self.actions:
            economics = ECONOMICS_BY_ACTION[action.key]
            estimated_revenue = realized_price(action, scenario) * 0.8 * state.acres
            score = estimated_revenue - economics.operating_cost_per_acre * state.acres
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
