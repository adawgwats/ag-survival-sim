from __future__ import annotations

from pathlib import Path

import pytest

from ag_survival_sim import (
    Action,
    FarmState,
    WeatherTransform,
    apply_weather_transform,
    build_iowa_maize_crop_model,
    kg_per_hectare_to_hundredweight_per_acre,
    kg_per_hectare_to_short_tons_per_acre,
    kg_per_hectare_to_bushels_per_acre,
    resolve_dssat_root,
)
from ag_survival_sim.scenario import AnnualScenario


def test_kg_per_hectare_to_bushels_per_acre_converts_corn_units() -> None:
    converted = kg_per_hectare_to_bushels_per_acre(6277.0, crop="corn")
    assert converted == pytest.approx(100.0, rel=0.01)


def test_kg_per_hectare_to_hundredweight_per_acre_converts_units() -> None:
    converted = kg_per_hectare_to_hundredweight_per_acre(4484.0)
    assert converted == pytest.approx(40.0, rel=0.02)


def test_kg_per_hectare_to_short_tons_per_acre_converts_units() -> None:
    converted = kg_per_hectare_to_short_tons_per_acre(4484.0)
    assert converted == pytest.approx(2.0, rel=0.02)


def test_apply_weather_transform_changes_daily_values(tmp_path: Path) -> None:
    weather_path = tmp_path / "TEST.WTH"
    weather_path.write_text(
        "\n".join(
            [
                "*WEATHER: TEST",
                "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT  CO2",
                "  TEST   42.000  -93.000   300   9.1  15.5  1.5   3.00  365",
                "@DATE  SRAD  TMAX  TMIN  RAIN",
                "99001   10.0  20.0  10.0   5.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    apply_weather_transform(
        weather_path,
        weather_path,
        WeatherTransform(
            rain_multiplier=0.5,
            srad_multiplier=1.1,
            tmax_delta_c=2.0,
            tmin_delta_c=1.0,
        ),
    )

    lines = weather_path.read_text(encoding="utf-8").splitlines()
    assert lines[-1].split() == ["99001", "11.0", "22.0", "11.0", "2.5"]


@pytest.mark.integration
def test_real_iowa_maize_crop_model_orders_weather_regimes(tmp_path: Path) -> None:
    try:
        resolve_dssat_root()
    except FileNotFoundError:
        pytest.skip("real DSSAT installation not available")

    state = FarmState.initial()
    action = Action("corn", "medium")
    model = build_iowa_maize_crop_model(workspace_root=tmp_path / "iowa_maize")

    yields = {}
    for regime in ("good", "normal", "drought"):
        scenario = AnnualScenario(
            year_index=0,
            weather_regime=regime,
            weather_yield_multiplier=1.0,
            market_price_multiplier=1.0,
            operating_cost_multiplier=1.0,
        )
        yields[regime] = model.yield_per_acre(state=state, action=action, scenario=scenario)

    assert yields["good"] > yields["normal"] > yields["drought"]
