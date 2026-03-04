from ag_survival_sim import (
    Action,
    AllocationSlice,
    ChristensenKnightianPortfolioPolicy,
    FarmState,
    GreedyMarginPortfolioPolicy,
    LearnedPortfolioConfig,
    PortfolioAllocation,
    RandomPortfolioPolicy,
    ScenarioGenerator,
    StaticPortfolioPolicy,
    TableCropModel,
    build_learning_exploration_policies,
    evaluate_portfolio_policies,
    train_learned_rollout_portfolio_policy,
)
from ag_survival_sim.portfolio_learning import PortfolioCandidateGenerator
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
    assert 0.0 <= summary.metrics["split"].full_horizon_survival_rate <= 1.0
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


def test_random_portfolio_policy_returns_feasible_allocation() -> None:
    simulator = build_portfolio_simulator()
    candidate_generator = PortfolioCandidateGenerator(
        actions=(
            Action("corn", "irrigated_low"),
            Action("soy", "medium"),
            Action("peanut", "medium"),
        ),
        crop_model=simulator.crop_model,
        top_action_count=3,
        random_samples=8,
        max_active_actions=3,
    )
    policy = RandomPortfolioPolicy(candidate_generator=candidate_generator, seed=11)
    state = FarmState.initial(cash=250_000.0, debt=0.0, credit_limit=50_000.0, acres=100.0)
    scenario = ScenarioGenerator(seed=4).generate_path(1)[0]
    allocation = policy.choose_allocation(state, scenario)

    assert allocation.total_acres <= 100.0 + 1e-6
    assert allocation.total_acres > 0.0


def test_learned_rollout_policy_trains_and_returns_feasible_allocation() -> None:
    simulator = build_portfolio_simulator()
    actions = (
        Action("corn", "irrigated_low"),
        Action("soy", "medium"),
        Action("peanut", "medium"),
    )
    learned_policy = train_learned_rollout_portfolio_policy(
        actions=actions,
        crop_model=simulator.crop_model,
        initial_state=FarmState.initial(cash=250_000.0, debt=0.0, credit_limit=100_000.0, acres=100.0),
        config=LearnedPortfolioConfig(
            horizon_years=4,
            training_paths=2,
            training_seed=17,
            random_exploration_policies=1,
            epochs=80,
            candidate_random_samples=6,
            top_action_count=3,
            bankruptcy_penalty_per_acre=2_000.0,
        ),
        exploration_policies=build_learning_exploration_policies(
            actions=actions,
            crop_model=simulator.crop_model,
        ),
    )
    scenario = ScenarioGenerator(seed=6).generate_path(1)[0]
    allocation = learned_policy.choose_allocation(
        FarmState.initial(cash=250_000.0, debt=0.0, credit_limit=100_000.0, acres=100.0),
        scenario,
    )

    assert learned_policy.training_summary.example_count > 0
    assert allocation.total_acres <= 100.0 + 1e-6
    assert allocation.total_acres > 0.0
