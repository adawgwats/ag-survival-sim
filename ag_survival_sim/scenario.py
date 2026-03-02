from __future__ import annotations

import random
from dataclasses import dataclass


WEATHER_REGIMES = ("good", "normal", "drought")


@dataclass(frozen=True)
class AnnualScenario:
    year_index: int
    weather_regime: str
    weather_yield_multiplier: float
    market_price_multiplier: float
    operating_cost_multiplier: float
    basis_penalty: float = 0.0


class ScenarioGenerator:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate_path(self, horizon_years: int, *, path_index: int = 0) -> list[AnnualScenario]:
        rng = random.Random((self.seed, path_index).__hash__())
        path: list[AnnualScenario] = []
        previous_weather = "normal"

        for year_index in range(horizon_years):
            weather_regime = self._sample_weather(rng, previous_weather)
            previous_weather = weather_regime
            path.append(
                AnnualScenario(
                    year_index=year_index,
                    weather_regime=weather_regime,
                    weather_yield_multiplier=self._yield_multiplier(weather_regime),
                    market_price_multiplier=self._price_multiplier(rng, weather_regime),
                    operating_cost_multiplier=self._operating_cost_multiplier(rng, weather_regime),
                    basis_penalty=self._basis_penalty(weather_regime),
                )
            )
        return path

    @staticmethod
    def _sample_weather(rng: random.Random, previous_weather: str) -> str:
        roll = rng.random()
        if previous_weather == "drought":
            if roll < 0.30:
                return "drought"
            if roll < 0.75:
                return "normal"
            return "good"
        if previous_weather == "good":
            if roll < 0.20:
                return "good"
            if roll < 0.80:
                return "normal"
            return "drought"
        if roll < 0.15:
            return "good"
        if roll < 0.80:
            return "normal"
        return "drought"

    @staticmethod
    def _yield_multiplier(weather_regime: str) -> float:
        return {
            "good": 1.08,
            "normal": 1.0,
            "drought": 0.72,
        }[weather_regime]

    @staticmethod
    def _price_multiplier(rng: random.Random, weather_regime: str) -> float:
        base = {
            "good": 0.95,
            "normal": 1.0,
            "drought": 1.15,
        }[weather_regime]
        return max(base + rng.uniform(-0.08, 0.08), 0.2)

    @staticmethod
    def _operating_cost_multiplier(rng: random.Random, weather_regime: str) -> float:
        base = {
            "good": 0.98,
            "normal": 1.0,
            "drought": 1.06,
        }[weather_regime]
        return max(base + rng.uniform(-0.03, 0.05), 0.1)

    @staticmethod
    def _basis_penalty(weather_regime: str) -> float:
        return {
            "good": 0.02,
            "normal": 0.04,
            "drought": 0.08,
        }[weather_regime]
