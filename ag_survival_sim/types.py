from __future__ import annotations

from dataclasses import dataclass, replace


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
        year: int = 0,
    ) -> "FarmState":
        return cls(
            cash=cash,
            debt=debt,
            credit_limit=credit_limit,
            acres=acres,
            year=year,
            alive=True,
        )

    @property
    def remaining_credit(self) -> float:
        return max(self.credit_limit - max(self.debt, 0.0), 0.0)

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
