from ag_survival_sim import Action, FarmState, TableCropModel
from ag_survival_sim.scenario import AnnualScenario


def test_table_crop_model_returns_weather_specific_yield() -> None:
    model = TableCropModel.from_records(
        [
            ("corn", "low", "normal", 165.0),
            ("corn", "low", "drought", 105.0),
        ]
    )

    normal = model.yield_per_acre(
        state=FarmState.initial(),
        action=Action("corn", "low"),
        scenario=AnnualScenario(
            year_index=0,
            weather_regime="normal",
            weather_yield_multiplier=1.0,
            market_price_multiplier=1.0,
            operating_cost_multiplier=1.0,
        ),
    )
    drought = model.yield_per_acre(
        state=FarmState.initial(),
        action=Action("corn", "low"),
        scenario=AnnualScenario(
            year_index=0,
            weather_regime="drought",
            weather_yield_multiplier=1.0,
            market_price_multiplier=1.0,
            operating_cost_multiplier=1.0,
        ),
    )

    assert normal > drought
