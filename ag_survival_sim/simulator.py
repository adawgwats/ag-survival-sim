from __future__ import annotations

from dataclasses import dataclass

from .crop_model import CropModel
from .finance import (
    ANNUAL_INTEREST_RATE,
    NEXT_YEAR_OPERATING_BUFFER,
    debt_payment,
    dscr,
    liquidity_failure,
    operating_cost,
    realized_price,
)
from .scenario import AnnualScenario
from .types import Action, FarmState, FarmStepRecord


class FarmSimulator:
    def __init__(self, crop_model: CropModel) -> None:
        self.crop_model = crop_model

    def step(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> FarmStepRecord:
        if not state.alive:
            raise ValueError("cannot step a dead farm state.")

        yield_per_acre = self.crop_model.yield_per_acre(
            state=state,
            action=action,
            scenario=scenario,
        )
        price = realized_price(action, scenario)
        gross_revenue = yield_per_acre * state.acres * price
        total_operating_cost = operating_cost(action, state.acres, scenario)
        annual_debt_payment = debt_payment(state)
        net_income = gross_revenue - total_operating_cost - annual_debt_payment
        next_cash = state.cash + net_income
        next_debt = max(state.debt * (1.0 + ANNUAL_INTEREST_RATE) - annual_debt_payment, 0.0)
        coverage = dscr(net_income, annual_debt_payment)
        next_failures = state.consecutive_dscr_failures + 1 if coverage < 1.0 else 0

        next_year_min_operating_cost = total_operating_cost * NEXT_YEAR_OPERATING_BUFFER
        failed = liquidity_failure(
            ending_cash=next_cash,
            ending_debt=next_debt,
            credit_limit=state.credit_limit,
            next_year_min_operating_cost=next_year_min_operating_cost,
        ) or next_failures >= 2

        ending_state = state.advance_year(
            cash=next_cash,
            debt=next_debt,
            alive=not failed,
            consecutive_dscr_failures=next_failures,
            cumulative_profit=state.cumulative_profit + net_income,
        )

        return FarmStepRecord(
            starting_state=state,
            ending_state=ending_state,
            action=action,
            realized_yield_per_acre=yield_per_acre,
            realized_price=price,
            gross_revenue=gross_revenue,
            operating_cost=total_operating_cost,
            net_income=net_income,
            debt_payment=annual_debt_payment,
            dscr=coverage,
            weather_regime=scenario.weather_regime,
        )
