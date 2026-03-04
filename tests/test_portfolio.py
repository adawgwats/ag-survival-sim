from ag_survival_sim import (
    Action,
    AllocationSlice,
    ChristensenKnightianPortfolioPolicy,
    FarmState,
    GreedyMarginPortfolioPolicy,
    PortfolioAllocation,
    ScenarioGenerator,
    StaticPortfolioPolicy,
    TableCropModel,
    evaluate_portfolio_policies,
)
from ag_survival_sim.portfolio_simulator import PortfolioFarmSimulator


def build_portfolio_simulator() -> PortfolioFarmSimulator:
    return PortfolioFarmSimulator(
        crop_model=TableCropModel.from_records(
            [
                ("corn", "irrigated_low", "good", 180.0),
                ("corn", "irrigated_low", "normal", 160.0),
                ("corn", "irrigated_low", "drought", 120.0),
                ("soy", "medium", "good", 60.0),
                ("soy", "medium", "normal", 55.0),
                ("soy", "medium", "drought", 44.0),
                ("peanut", "medium", "good", 2.6),
                ("peanut", "medium", "normal", 2.2),
                ("peanut", "medium", "drought", 1.7),
            ]
        )
    )


def test_portfolio_allocation_rejects_duplicate_actions() -> None:
    try:
        PortfolioAllocation(
            (
                AllocationSlice(Action("corn", "irrigated_low"), 100.0),
                AllocationSlice(Action("corn", "irrigated_low"), 50.0),
            )
        )
    except ValueError:
        return
    raise AssertionError("expected duplicate portfolio allocation to raise ValueError")


def test_static_portfolio_policy_scales_to_capital_constraint() -> None:
    policy = StaticPortfolioPolicy(
        shares=(
            AllocationSlice(Action("corn", "irrigated_low"), 0.7),
            AllocationSlice(Action("soy", "medium"), 0.3),
        )
    )
    state = FarmState.initial(cash=40_000.0, debt=0.0, credit_limit=0.0, acres=100.0)
    allocation = policy.choose_allocation(state, ScenarioGenerator(seed=1).generate_path(1)[0])

    assert allocation.total_acres < 100.0
    assert allocation.total_acres > 0.0


def test_greedy_margin_policy_uses_multiple_actions_when_share_cap_is_binding() -> None:
    simulator = build_portfolio_simulator()
    policy = GreedyMarginPortfolioPolicy(
        actions=(
            Action("corn", "irrigated_low"),
            Action("soy", "medium"),
            Action("peanut", "medium"),
        ),
        crop_model=simulator.crop_model,
        max_share_per_action=0.5,
    )
    state = FarmState.initial(cash=500_000.0, debt=0.0, credit_limit=100_000.0, acres=100.0)
    scenario = ScenarioGenerator(seed=2).generate_path(1)[0]
    allocation = policy.choose_allocation(state, scenario)

    assert allocation.total_acres <= 100.0
    assert len(allocation.slices) >= 2


def test_portfolio_evaluation_returns_metrics() -> None:
    simulator = build_portfolio_simulator()
    summary = evaluate_portfolio_policies(
        simulator=simulator,
        scenario_generator=ScenarioGenerator(seed=5),
        policies={
            "split": StaticPortfolioPolicy(
                shares=(
                    AllocationSlice(Action("corn", "irrigated_low"), 0.5),
                    AllocationSlice(Action("soy", "medium"), 0.5),
                )
            ),
            "greedy": GreedyMarginPortfolioPolicy(
                actions=(
                    Action("corn", "irrigated_low"),
                    Action("soy", "medium"),
                    Action("peanut", "medium"),
                ),
                crop_model=simulator.crop_model,
                max_share_per_action=0.6,
            ),
        },
        initial_state=FarmState.initial(cash=200_000.0, debt=0.0, credit_limit=100_000.0, acres=100.0),
        horizon_years=4,
        num_paths=3,
    )

    assert set(summary.metrics) == {"split", "greedy"}
    assert summary.metrics["split"].mean_survival_years >= 0.0
    assert summary.metrics["greedy"].mean_cumulative_profit != 0.0


def test_christensen_knightian_policy_returns_feasible_diversified_allocation() -> None:
    simulator = build_portfolio_simulator()
    policy = ChristensenKnightianPortfolioPolicy(
        actions=(
            Action("corn", "irrigated_low"),
            Action("soy", "medium"),
            Action("peanut", "medium"),
        ),
        crop_model=simulator.crop_model,
        max_share_per_action=0.5,
        max_share_per_crop=0.7,
    )
    state = FarmState.initial(
        cash=500_000.0,
        debt=0.0,
        credit_limit=100_000.0,
        acres=100.0,
        land_mortgage_grace_years=0,
    )
    scenario = ScenarioGenerator(seed=3).generate_path(1)[0]
    allocation = policy.choose_allocation(state, scenario)

    assert allocation.total_acres <= 100.0
    assert allocation.total_acres > 0.0
    assert all(allocation_slice.acres <= 50.0 + 1e-6 for allocation_slice in allocation.slices)
