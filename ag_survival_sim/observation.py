from __future__ import annotations

import random
from dataclasses import dataclass

from .types import FarmStepRecord


@dataclass(frozen=True)
class ObservationRecord:
    observed_net_income: float | None
    observed_yield_per_acre: float | None
    observed_price: float | None
    fully_observed: bool


class SelectiveObservationRule:
    def __init__(self, *, seed: int = 0, distressed_penalty: float = 0.35) -> None:
        self.seed = seed
        self.distressed_penalty = distressed_penalty

    def observe(self, record: FarmStepRecord, *, path_index: int, step_index: int) -> ObservationRecord:
        rng = random.Random(hash((self.seed, path_index, step_index, record.starting_state.year)))
        probability = 0.95

        if record.net_income < 0.0:
            probability -= self.distressed_penalty
        if record.weather_regime == "drought":
            probability -= 0.10

        probability = min(max(probability, 0.05), 1.0)
        fully_observed = rng.random() < probability

        if fully_observed:
            return ObservationRecord(
                observed_net_income=record.net_income,
                observed_yield_per_acre=record.realized_yield_per_acre,
                observed_price=record.realized_price,
                fully_observed=True,
            )
        return ObservationRecord(
            observed_net_income=None,
            observed_yield_per_acre=None,
            observed_price=None,
            fully_observed=False,
        )


@dataclass(frozen=True)
class ObservationProcess:
    rule: SelectiveObservationRule

    def apply(
        self,
        records: list[FarmStepRecord],
        *,
        path_index: int,
    ) -> list[ObservationRecord]:
        return [
            self.rule.observe(record, path_index=path_index, step_index=step_index)
            for step_index, record in enumerate(records)
        ]
