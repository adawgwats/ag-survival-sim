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
    ("soy", "low"): CropEconomics(base_price=11.40, operating_cost_per_acre=310.0),
    ("soy", "medium"): CropEconomics(base_price=11.40, operating_cost_per_acre=380.0),
    ("wheat", "low"): CropEconomics(base_price=6.10, operating_cost_per_acre=250.0),
    ("wheat", "medium"): CropEconomics(base_price=6.10, operating_cost_per_acre=325.0),
}

ANNUAL_INTEREST_RATE = 0.06
MINIMUM_DEBT_PAYMENT_RATE = 0.10
NEXT_YEAR_OPERATING_BUFFER = 0.50


def realized_price(action: Action, scenario: AnnualScenario) -> float:
    economics = ECONOMICS_BY_ACTION[action.key]
    return max(economics.base_price * scenario.market_price_multiplier - scenario.basis_penalty, 0.01)


def operating_cost(action: Action, acres: float, scenario: AnnualScenario) -> float:
    economics = ECONOMICS_BY_ACTION[action.key]
    return economics.operating_cost_per_acre * acres * scenario.operating_cost_multiplier


def debt_payment(state: FarmState) -> float:
    return state.debt * (ANNUAL_INTEREST_RATE + MINIMUM_DEBT_PAYMENT_RATE)


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
