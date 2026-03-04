from __future__ import annotations

from dataclasses import dataclass, replace

DEFAULT_LAND_VALUE_PER_ACRE = 4_000.0
DEFAULT_LAND_MORTGAGE_RATE = 0.045
DEFAULT_LAND_MORTGAGE_YEARS = 30
DEFAULT_LAND_FINANCED_FRACTION = 0.5


@dataclass(frozen=True)
class Action:
    crop: str
    input_level: str

    @property
    def key(self) -> tuple[str, str]:
        return self.crop, self.input_level


@dataclass(frozen=True)
class FarmState:
    cash: float
    debt: float
    credit_limit: float
    acres: float
    land_value_per_acre: float
    land_mortgage_balance: float
    land_mortgage_rate: float
    land_mortgage_years_remaining: int
    year: int
    alive: bool
    consecutive_dscr_failures: int = 0
    cumulative_profit: float = 0.0

    @classmethod
    def initial(
        cls,
        *,
        cash: float = 200_000.0,
        debt: float = 150_000.0,
        credit_limit: float = 125_000.0,
        acres: float = 500.0,
        land_value_per_acre: float = DEFAULT_LAND_VALUE_PER_ACRE,
        land_financed_fraction: float = DEFAULT_LAND_FINANCED_FRACTION,
        land_mortgage_rate: float = DEFAULT_LAND_MORTGAGE_RATE,
        land_mortgage_years: int = DEFAULT_LAND_MORTGAGE_YEARS,
        year: int = 0,
    ) -> "FarmState":
        if not 0.0 <= land_financed_fraction <= 1.0:
            raise ValueError("land_financed_fraction must be in [0, 1].")
        if land_mortgage_years < 0:
            raise ValueError("land_mortgage_years must be nonnegative.")
        if land_mortgage_rate < 0.0:
            raise ValueError("land_mortgage_rate must be nonnegative.")
        land_mortgage_balance = acres * land_value_per_acre * land_financed_fraction
        return cls(
            cash=cash,
            debt=debt,
            credit_limit=credit_limit,
            acres=acres,
            land_value_per_acre=land_value_per_acre,
            land_mortgage_balance=land_mortgage_balance,
            land_mortgage_rate=land_mortgage_rate,
            land_mortgage_years_remaining=land_mortgage_years if land_mortgage_balance > 0.0 else 0,
            year=year,
            alive=True,
        )

    @property
    def remaining_credit(self) -> float:
        return max(self.credit_limit - max(self.debt, 0.0), 0.0)

    @property
    def land_value(self) -> float:
        return self.land_value_per_acre * self.acres

    def advance_year(self, **changes: float | int | bool) -> "FarmState":
        return replace(self, year=self.year + 1, **changes)


@dataclass(frozen=True)
class FarmStepRecord:
    starting_state: FarmState
    ending_state: FarmState
    action: Action
    realized_yield_per_acre: float
    realized_price: float
    gross_revenue: float
    operating_cost: float
    net_income: float
    debt_payment: float
    dscr: float
    weather_regime: str
