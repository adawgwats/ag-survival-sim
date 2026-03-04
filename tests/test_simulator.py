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


def test_unaffordable_action_triggers_failure() -> None:
    simulator = FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "irrigated_high", "normal", 200.0),
            ]
        )
    )
    record = simulator.step(
        state=FarmState.initial(cash=20_000.0, debt=150_000.0, credit_limit=30_000.0, acres=500.0),
        action=Action("corn", "irrigated_high"),
        scenario=build_scenario("normal"),
    )

    assert record.ending_state.alive is False


def test_land_mortgage_grace_period_delays_payments_without_burning_amortization_years() -> None:
    simulator = FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "low", "normal", 180.0),
            ]
        )
    )
    state = FarmState.initial(
        cash=500_000.0,
        debt=0.0,
        credit_limit=200_000.0,
        acres=100.0,
        land_value_per_acre=2_000.0,
        land_financed_fraction=1.0,
        land_mortgage_rate=0.05,
        land_mortgage_years=10,
        land_mortgage_grace_years=2,
    )
    scenario = build_scenario("normal")

    first = simulator.step(state=state, action=Action("corn", "low"), scenario=scenario)
    second = simulator.step(state=first.ending_state, action=Action("corn", "low"), scenario=scenario)
    third = simulator.step(state=second.ending_state, action=Action("corn", "low"), scenario=scenario)

    assert first.debt_payment == 0.0
    assert second.debt_payment == 0.0
    assert third.debt_payment > 0.0
    assert first.ending_state.land_mortgage_grace_years_remaining == 1
    assert second.ending_state.land_mortgage_grace_years_remaining == 0
    assert first.ending_state.land_mortgage_years_remaining == 10
    assert second.ending_state.land_mortgage_years_remaining == 10
    assert third.ending_state.land_mortgage_years_remaining == 9


def test_dscr_uses_pre_debt_operating_cash_flow() -> None:
    simulator = FarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("soy", "low", "drought", 38.0),
            ]
        )
    )
    state = FarmState.initial(
        cash=250_000.0,
        debt=0.0,
        credit_limit=100_000.0,
        acres=100.0,
        land_value_per_acre=4_000.0,
        land_financed_fraction=0.5,
        land_mortgage_rate=0.045,
        land_mortgage_years=30,
        land_mortgage_grace_years=0,
    )
    record = simulator.step(
        state=state,
        action=Action("soy", "low"),
        scenario=build_scenario("drought"),
    )

    assert record.debt_payment > 0.0
    assert record.net_income > 0.0
    assert record.dscr > 1.0
