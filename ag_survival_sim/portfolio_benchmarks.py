from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .crop_model import CropModel
from .dssat_benchmarks import build_benchmark_crop_model
from .portfolio import (
    AllocationSlice,
    ChristensenKnightianPortfolioPolicy,
    GreedyMarginPortfolioPolicy,
    PortfolioPolicy,
    StaticPortfolioPolicy,
)
from .portfolio_simulator import PortfolioFarmSimulator
from .types import Action


@dataclass(frozen=True)
class PortfolioOption:
    action: Action
    source_benchmark_name: str


@dataclass(frozen=True)
class PortfolioBenchmarkDefinition:
    name: str
    options: tuple[PortfolioOption, ...]
    description: str = ""


PORTFOLIO_BENCHMARK_DEFINITIONS: dict[str, PortfolioBenchmarkDefinition] = {
    "georgia_diversified_portfolio": PortfolioBenchmarkDefinition(
        name="georgia_diversified_portfolio",
        options=(
            PortfolioOption(Action("corn", "rainfed_low"), "georgia_maize_management"),
            PortfolioOption(Action("corn", "rainfed_high"), "georgia_maize_management"),
            PortfolioOption(Action("corn", "irrigated_low"), "georgia_maize_management"),
            PortfolioOption(Action("corn", "irrigated_high"), "georgia_maize_management"),
            PortfolioOption(Action("soy", "low"), "georgia_soybean"),
            PortfolioOption(Action("soy", "medium"), "georgia_soybean"),
            PortfolioOption(Action("peanut", "low"), "georgia_peanut"),
            PortfolioOption(Action("peanut", "medium"), "georgia_peanut"),
        ),
        description="Georgia diversified benchmark spanning maize, soybean, and peanut.",
    ),
}


class CompositeBenchmarkCropModel:
    def __init__(
        self,
        *,
        models_by_benchmark: Mapping[str, CropModel],
        benchmark_name_by_action: Mapping[tuple[str, str], str],
    ) -> None:
        self._models_by_benchmark = dict(models_by_benchmark)
        self._benchmark_name_by_action = dict(benchmark_name_by_action)
        self._cache: dict[tuple[str, str, str, str], float] = {}

    def yield_per_acre(self, *, state, action, scenario) -> float:
        benchmark_name = self._benchmark_name_by_action[action.key]
        cache_key = (benchmark_name, action.crop, action.input_level, scenario.weather_regime)
        if cache_key not in self._cache:
            crop_model = self._models_by_benchmark[benchmark_name]
            self._cache[cache_key] = crop_model.yield_per_acre(
                state=state,
                action=action,
                scenario=scenario,
            )
        return self._cache[cache_key]


def list_portfolio_benchmark_definitions() -> tuple[PortfolioBenchmarkDefinition, ...]:
    return tuple(
        PORTFOLIO_BENCHMARK_DEFINITIONS[name]
        for name in sorted(PORTFOLIO_BENCHMARK_DEFINITIONS)
    )


def get_portfolio_benchmark_definition(name: str) -> PortfolioBenchmarkDefinition:
    try:
        return PORTFOLIO_BENCHMARK_DEFINITIONS[name]
    except KeyError as error:
        raise KeyError(f"unknown portfolio benchmark '{name}'") from error


def build_portfolio_benchmark_crop_model(
    benchmark_name: str,
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/portfolio_benchmark",
) -> CompositeBenchmarkCropModel:
    definition = get_portfolio_benchmark_definition(benchmark_name)
    unique_source_benchmarks = sorted({option.source_benchmark_name for option in definition.options})
    models_by_benchmark = {
        source_benchmark_name: build_benchmark_crop_model(
            source_benchmark_name,
            dssat_root=dssat_root,
            workspace_root=Path(workspace_root) / source_benchmark_name,
        )
        for source_benchmark_name in unique_source_benchmarks
    }
    benchmark_name_by_action = {
        option.action.key: option.source_benchmark_name
        for option in definition.options
    }
    return CompositeBenchmarkCropModel(
        models_by_benchmark=models_by_benchmark,
        benchmark_name_by_action=benchmark_name_by_action,
    )


def build_portfolio_benchmark_simulator(
    benchmark_name: str,
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/portfolio_benchmark",
) -> PortfolioFarmSimulator:
    return PortfolioFarmSimulator(
        crop_model=build_portfolio_benchmark_crop_model(
            benchmark_name,
            dssat_root=dssat_root,
            workspace_root=workspace_root,
        )
    )


def build_portfolio_demo_policies(
    benchmark_name: str,
    *,
    crop_model: CropModel,
) -> dict[str, PortfolioPolicy]:
    definition = get_portfolio_benchmark_definition(benchmark_name)
    actions = tuple(option.action for option in definition.options)

    return {
        "equal_mix": StaticPortfolioPolicy(
            shares=(
                AllocationSlice(Action("corn", "irrigated_low"), 1.0 / 3.0),
                AllocationSlice(Action("soy", "medium"), 1.0 / 3.0),
                AllocationSlice(Action("peanut", "medium"), 1.0 / 3.0),
            )
        ),
        "cash_buffer_mix": StaticPortfolioPolicy(
            shares=(
                AllocationSlice(Action("corn", "rainfed_low"), 0.40),
                AllocationSlice(Action("soy", "medium"), 0.35),
                AllocationSlice(Action("peanut", "low"), 0.25),
            )
        ),
        "greedy_margin": GreedyMarginPortfolioPolicy(
            actions=actions,
            crop_model=crop_model,
            max_share_per_action=0.6,
        ),
        "christensen_knightian": ChristensenKnightianPortfolioPolicy(
            actions=actions,
            crop_model=crop_model,
            max_share_per_action=0.5,
            max_share_per_crop=0.7,
        ),
    }
