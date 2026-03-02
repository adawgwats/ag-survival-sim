from ag_survival_sim import ObservationProcess, SelectiveObservationRule
from ag_survival_sim.types import Action, FarmState, FarmStepRecord


def test_selective_observation_can_hide_distressed_outcomes() -> None:
    rule = SelectiveObservationRule(seed=7, distressed_penalty=0.8)
    process = ObservationProcess(rule)
    record = FarmStepRecord(
        starting_state=FarmState.initial(),
        ending_state=FarmState.initial(cash=100_000.0, debt=200_000.0),
        action=Action("corn", "low"),
        realized_yield_per_acre=90.0,
        realized_price=4.0,
        gross_revenue=180_000.0,
        operating_cost=220_000.0,
        net_income=-60_000.0,
        debt_payment=24_000.0,
        dscr=-2.5,
        weather_regime="drought",
    )

    observations = [
        process.apply([record], path_index=path_index)[0]
        for path_index in range(20)
    ]

    assert len(observations) == 20
    assert any(observation.fully_observed is False for observation in observations)
