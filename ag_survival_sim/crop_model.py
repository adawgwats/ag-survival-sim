from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .scenario import AnnualScenario
from .types import Action, FarmState


class CropModel(Protocol):
    def yield_per_acre(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> float:
        ...


@dataclass(frozen=True)
class DSSATRecord:
    crop: str
    input_level: str
    weather_regime: str
    yield_per_acre: float


class TableCropModel:
    def __init__(self, records: list[DSSATRecord]) -> None:
        if not records:
            raise ValueError("at least one DSSAT record is required.")
        self._table = {
            (record.crop, record.input_level, record.weather_regime): record.yield_per_acre
            for record in records
        }

    @classmethod
    def from_records(
        cls,
        records: list[tuple[str, str, str, float]],
    ) -> "TableCropModel":
        return cls(
            [
                DSSATRecord(
                    crop=crop,
                    input_level=input_level,
                    weather_regime=weather_regime,
                    yield_per_acre=yield_per_acre,
                )
                for crop, input_level, weather_regime, yield_per_acre in records
            ]
        )

    def yield_per_acre(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> float:
        del state
        try:
            base_yield = self._table[(action.crop, action.input_level, scenario.weather_regime)]
        except KeyError as error:
            raise KeyError(
                "missing DSSAT-style record for "
                f"{action.crop}/{action.input_level}/{scenario.weather_regime}"
            ) from error
        return base_yield * scenario.weather_yield_multiplier
