from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .observation import ObservationProcess
from .policy import FarmPolicy
from .scenario import ScenarioGenerator
from .simulator import FarmSimulator
from .types import FarmState


@dataclass(frozen=True)
class TrainingExample:
    path_index: int
    step_index: int
    year: int
    crop: str
    input_level: str
    weather_regime: str
    cash: float
    debt: float
    credit_limit: float
    acres: float
    land_value_per_acre: float
    land_mortgage_balance: float
    land_mortgage_rate: float
    land_mortgage_years_remaining: int
    land_mortgage_grace_years_remaining: int
    latent_yield_per_acre: float
    latent_net_income: float
    latent_price: float
    observed_yield_per_acre: float | None
    observed_net_income: float | None
    observed_price: float | None
    label_observed: bool
    group_id: str
    farm_alive_next_year: bool


def default_group_id(example: TrainingExample) -> str:
    if example.weather_regime == "drought" or example.latent_net_income < 0.0:
        return "distressed"
    return "stable"


def generate_training_examples(
    *,
    simulator: FarmSimulator,
    scenario_generator: ScenarioGenerator,
    policy: FarmPolicy,
    observation_process: ObservationProcess,
    initial_state: FarmState,
    horizon_years: int,
    num_paths: int,
    group_id_fn: Callable[[TrainingExample], str] = default_group_id,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    for path_index in range(num_paths):
        path = scenario_generator.generate_path(horizon_years, path_index=path_index)
        state = initial_state
        path_records = []
        for scenario in path:
            if not state.alive:
                break
            action = policy.choose_action(state, scenario)
            record = simulator.step(state=state, action=action, scenario=scenario)
            path_records.append(record)
            state = record.ending_state

        observations = observation_process.apply(path_records, path_index=path_index)
        for step_index, (record, observed) in enumerate(zip(path_records, observations)):
            base_example = TrainingExample(
                path_index=path_index,
                step_index=step_index,
                year=record.starting_state.year,
                crop=record.action.crop,
                input_level=record.action.input_level,
                weather_regime=record.weather_regime,
                cash=record.starting_state.cash,
                debt=record.starting_state.debt,
                credit_limit=record.starting_state.credit_limit,
                acres=record.starting_state.acres,
                land_value_per_acre=record.starting_state.land_value_per_acre,
                land_mortgage_balance=record.starting_state.land_mortgage_balance,
                land_mortgage_rate=record.starting_state.land_mortgage_rate,
                land_mortgage_years_remaining=record.starting_state.land_mortgage_years_remaining,
                land_mortgage_grace_years_remaining=record.starting_state.land_mortgage_grace_years_remaining,
                latent_yield_per_acre=record.realized_yield_per_acre,
                latent_net_income=record.net_income,
                latent_price=record.realized_price,
                observed_yield_per_acre=observed.observed_yield_per_acre,
                observed_net_income=observed.observed_net_income,
                observed_price=observed.observed_price,
                label_observed=observed.fully_observed,
                group_id="",
                farm_alive_next_year=record.ending_state.alive,
            )
            examples.append(
                replace(
                    base_example,
                    group_id=group_id_fn(base_example),
                )
            )

    return examples
