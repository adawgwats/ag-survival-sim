from __future__ import annotations

from dataclasses import dataclass

from .scenario import AnnualScenario
from .types import Action, FarmState


@dataclass(frozen=True)
class CropEconomics:
    base_price: float
    operating_cost_per_acre: float


ECONOMICS_BY_ACTION: dict[tuple[str, str], CropEconomics] = {
    ("corn", "low"): CropEconomics(base_price=4.70, operating_cost_per_acre=520.0),
    ("corn", "medium"): CropEconomics(base_price=4.70, operating_cost_per_acre=620.0),
    ("corn", "rainfed_low"): CropEconomics(base_price=4.70, operating_cost_per_acre=520.0),
    ("corn", "rainfed_high"): CropEconomics(base_price=4.70, operating_cost_per_acre=620.0),
    ("corn", "irrigated_low"): CropEconomics(base_price=4.70, operating_cost_per_acre=760.0),
    ("corn", "irrigated_high"): CropEconomics(base_price=4.70, operating_cost_per_acre=900.0),
    ("soy", "low"): CropEconomics(base_price=11.40, operating_cost_per_acre=310.0),
    ("soy", "medium"): CropEconomics(base_price=11.40, operating_cost_per_acre=380.0),
    ("wheat", "low"): CropEconomics(base_price=6.10, operating_cost_per_acre=250.0),
    ("wheat", "medium"): CropEconomics(base_price=6.10, operating_cost_per_acre=325.0),
    ("rice", "low"): CropEconomics(base_price=16.0, operating_cost_per_acre=430.0),
    ("rice", "medium"): CropEconomics(base_price=16.0, operating_cost_per_acre=520.0),
    ("peanut", "low"): CropEconomics(base_price=525.0, operating_cost_per_acre=360.0),
    ("peanut", "medium"): CropEconomics(base_price=525.0, operating_cost_per_acre=440.0),
    ("sunflower", "low"): CropEconomics(base_price=22.0, operating_cost_per_acre=210.0),
    ("sunflower", "medium"): CropEconomics(base_price=22.0, operating_cost_per_acre=295.0),
}

ANNUAL_INTEREST_RATE = 0.06
MINIMUM_DEBT_PAYMENT_RATE = 0.10
NEXT_YEAR_OPERATING_BUFFER = 0.50


def realized_price(action: Action, scenario: AnnualScenario) -> float:
    economics = ECONOMICS_BY_ACTION[action.key]
    return max(economics.base_price * scenario.market_price_multiplier - scenario.basis_penalty, 0.01)


def planned_operating_cost(action: Action, acres: float) -> float:
    economics = ECONOMICS_BY_ACTION[action.key]
    return economics.operating_cost_per_acre * acres


def operating_cost(action: Action, acres: float, scenario: AnnualScenario) -> float:
    economics = ECONOMICS_BY_ACTION[action.key]
    return economics.operating_cost_per_acre * acres * scenario.operating_cost_multiplier


def amortized_payment(principal: float, annual_rate: float, years_remaining: int) -> float:
    if principal <= 0.0 or years_remaining <= 0:
        return 0.0
    if annual_rate <= 0.0:
        return principal / years_remaining
    growth = (1.0 + annual_rate) ** years_remaining
    return principal * annual_rate * growth / max(growth - 1.0, 1e-9)


def operating_debt_payment(state: FarmState) -> float:
    return state.debt * (ANNUAL_INTEREST_RATE + MINIMUM_DEBT_PAYMENT_RATE)


def land_mortgage_payment(state: FarmState) -> float:
    if state.land_mortgage_grace_years_remaining > 0:
        return 0.0
    return amortized_payment(
        state.land_mortgage_balance,
        state.land_mortgage_rate,
        state.land_mortgage_years_remaining,
    )


def debt_payment(state: FarmState) -> float:
    return operating_debt_payment(state) + land_mortgage_payment(state)


def next_land_mortgage_balance(state: FarmState, annual_payment: float) -> float:
    if state.land_mortgage_balance <= 0.0 or state.land_mortgage_years_remaining <= 0:
        return 0.0
    next_balance = state.land_mortgage_balance * (1.0 + state.land_mortgage_rate) - annual_payment
    return max(next_balance, 0.0)


def dscr(net_income: float, annual_debt_payment: float) -> float:
    if annual_debt_payment <= 0.0:
        return float("inf")
    return net_income / annual_debt_payment


def liquidity_failure(
    ending_cash: float,
    ending_debt: float,
    credit_limit: float,
    next_year_min_operating_cost: float,
) -> bool:
    remaining_credit = max(credit_limit - max(ending_debt, 0.0), 0.0)
    return ending_cash + remaining_credit < next_year_min_operating_cost
