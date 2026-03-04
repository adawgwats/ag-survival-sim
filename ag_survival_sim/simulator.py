from __future__ import annotations

from dataclasses import dataclass

from .crop_model import CropModel
from .finance import (
    ANNUAL_INTEREST_RATE,
    NEXT_YEAR_OPERATING_BUFFER,
    debt_payment,
    dscr,
    land_mortgage_payment,
    liquidity_failure,
    next_land_mortgage_balance,
    operating_debt_payment,
    operating_cost,
    planned_operating_cost,
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
        planned_cost = planned_operating_cost(action, state.acres)
        total_operating_cost = operating_cost(action, state.acres, scenario)
        annual_operating_debt_payment = operating_debt_payment(state)
        annual_land_payment = land_mortgage_payment(state)
        annual_debt_payment = debt_payment(state)
        operating_cash_flow = gross_revenue - total_operating_cost
        net_income = operating_cash_flow - annual_debt_payment
        next_cash = state.cash + net_income
        next_debt = max(state.debt * (1.0 + ANNUAL_INTEREST_RATE) - annual_operating_debt_payment, 0.0)
        next_land_balance = next_land_mortgage_balance(state, annual_land_payment)
        if next_land_balance > 0.0:
            next_land_grace_years = max(state.land_mortgage_grace_years_remaining - 1, 0)
            next_land_years = (
                state.land_mortgage_years_remaining
                if state.land_mortgage_grace_years_remaining > 0
                else max(state.land_mortgage_years_remaining - 1, 0)
            )
        else:
            next_land_years = 0
            next_land_grace_years = 0
        # Debt service coverage should be based on cash flow available to service debt,
        # not income after debt service has already been subtracted.
        coverage = dscr(operating_cash_flow, annual_debt_payment)
        next_failures = state.consecutive_dscr_failures + 1 if coverage < 1.0 else 0

        next_year_min_operating_cost = total_operating_cost * NEXT_YEAR_OPERATING_BUFFER
        financing_failure = planned_cost > state.cash + state.remaining_credit
        failed = financing_failure or liquidity_failure(
            ending_cash=next_cash,
            ending_debt=next_debt,
            credit_limit=state.credit_limit,
            next_year_min_operating_cost=next_year_min_operating_cost,
        ) or next_failures >= 2

        ending_state = state.advance_year(
            cash=next_cash,
            debt=next_debt,
            land_mortgage_balance=next_land_balance,
            land_mortgage_years_remaining=next_land_years,
            land_mortgage_grace_years_remaining=next_land_grace_years,
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
