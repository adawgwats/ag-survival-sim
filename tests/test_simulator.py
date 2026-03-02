from ag_survival_sim import Action, FarmSimulator, FarmState, TableCropModel
from ag_survival_sim.scenario import AnnualScenario


def build_scenario(weather_regime: str) -> AnnualScenario:
    return AnnualScenario(
        year_index=0,
        weather_regime=weather_regime,
        weather_yield_multiplier=1.0,
        market_price_multiplier=1.0,
        operating_cost_multiplier=1.0,
    )


def test_drought_reduces_next_state_financials() -> None:
    simulator = FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "low", "normal", 165.0),
                ("corn", "low", "drought", 105.0),
            ]
        )
    )
    state = FarmState.initial()
    normal_record = simulator.step(
        state=state,
        action=Action("corn", "low"),
        scenario=build_scenario("normal"),
    )
    drought_record = simulator.step(
        state=state,
        action=Action("corn", "low"),
        scenario=build_scenario("drought"),
    )

    assert normal_record.net_income > drought_record.net_income
    assert normal_record.ending_state.cash > drought_record.ending_state.cash
