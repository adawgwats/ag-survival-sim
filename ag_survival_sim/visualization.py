from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .evaluation import PolicyEvaluation


def _action_name(step) -> str:
    return f"{step.action.crop}_{step.action.input_level}"


def _import_pyplot(*, show: bool):
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise ImportError(
            "visualization helpers require matplotlib. "
            "Install ag-survival-sim with matplotlib available."
        ) from error
    return plt


def _build_title_lines(
    *,
    default_title: str,
    title: str | None,
    subtitle_lines: Sequence[str] | None,
) -> list[str]:
    title_lines = [title or default_title]
    if subtitle_lines:
        title_lines.extend(subtitle_lines)
    return title_lines


def plot_policy_action_traces(
    *,
    policy_evaluations: Mapping[str, PolicyEvaluation],
    path_index: int,
    output_path: str | Path,
    title: str | None = None,
    subtitle_lines: Sequence[str] | None = None,
    show: bool = False,
) -> Path:
    plt = _import_pyplot(show=show)

    if not policy_evaluations:
        raise ValueError("policy_evaluations must not be empty.")

    selected_names = list(policy_evaluations)
    missing_paths = [
        policy_name
        for policy_name, evaluation in policy_evaluations.items()
        if path_index < 0 or path_index >= len(evaluation.path_results)
    ]
    if missing_paths:
        raise IndexError(
            f"path_index={path_index} is out of range for policies: {', '.join(sorted(missing_paths))}"
        )

    action_names: list[str] = []
    for evaluation in policy_evaluations.values():
        for step in evaluation.path_results[path_index].steps:
            action_name = _action_name(step)
            if action_name not in action_names:
                action_names.append(action_name)
    if not action_names:
        raise ValueError("selected path has no recorded actions to plot.")

    action_index = {action_name: index for index, action_name in enumerate(action_names)}
    regime_steps = policy_evaluations[selected_names[0]].path_results[path_index].steps
    regimes = [step.weather_regime for step in regime_steps]
    max_step_count = max(
        len(evaluation.path_results[path_index].steps)
        for evaluation in policy_evaluations.values()
    )
    x_values = list(range(max_step_count))

    figure_height = max(3.5, 1.8 * len(selected_names) + 1.2)
    fig, axes = plt.subplots(
        len(selected_names) + 1,
        1,
        figsize=(12, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [0.5] + [1.0] * len(selected_names)},
    )
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    regime_axis = axes[0]
    regime_axis.set_yticks([])
    regime_axis.set_ylabel("weather")
    regime_axis.set_xlim(-0.25, max(max_step_count - 0.75, 0.75))
    for step_index, regime in enumerate(regimes):
        regime_axis.axvspan(step_index - 0.5, step_index + 0.5, color="#d9e2f3", alpha=0.45)
        regime_axis.text(step_index, 0.5, regime, ha="center", va="center", fontsize=8)
    regime_axis.set_frame_on(False)

    for axis, policy_name in zip(axes[1:], selected_names):
        path = policy_evaluations[policy_name].path_results[path_index]
        steps = path.steps
        xs = [step.starting_state.year for step in steps]
        ys = [action_index[_action_name(step)] for step in steps]

        axis.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
        if steps and not steps[-1].ending_state.alive:
            axis.scatter(
                [xs[-1]],
                [ys[-1]],
                marker="x",
                s=80,
                linewidths=2.0,
                color="#d62728",
                label="bankrupt",
            )
        axis.set_ylabel(policy_name)
        axis.set_yticks(range(len(action_names)))
        axis.set_yticklabels(action_names)
        axis.grid(axis="x", alpha=0.3)
        terminal_state = path.final_state
        axis.text(
            1.01,
            0.5,
            (
                f"survival={path.survival_years}\n"
                f"alive={terminal_state.alive}\n"
                f"cash={terminal_state.cash:,.0f}"
            ),
            transform=axis.transAxes,
            va="center",
            fontsize=8,
        )

    axes[-1].set_xlabel("simulation year")
    axes[-1].set_xticks(x_values)

    title_lines = _build_title_lines(
        default_title=f"policy action trace (path {path_index})",
        title=title,
        subtitle_lines=subtitle_lines,
    )
    fig.suptitle("\n".join(title_lines), fontsize=11, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_policy_profit_traces(
    *,
    policy_evaluations: Mapping[str, PolicyEvaluation],
    path_index: int,
    output_path: str | Path,
    title: str | None = None,
    subtitle_lines: Sequence[str] | None = None,
    show: bool = False,
) -> Path:
    plt = _import_pyplot(show=show)

    if not policy_evaluations:
        raise ValueError("policy_evaluations must not be empty.")

    selected_names = list(policy_evaluations)
    missing_paths = [
        policy_name
        for policy_name, evaluation in policy_evaluations.items()
        if path_index < 0 or path_index >= len(evaluation.path_results)
    ]
    if missing_paths:
        raise IndexError(
            f"path_index={path_index} is out of range for policies: {', '.join(sorted(missing_paths))}"
        )

    fig, (weather_axis, profit_axis, info_axis) = plt.subplots(
        3,
        1,
        figsize=(13, 8.5),
        gridspec_kw={"height_ratios": [0.5, 3.0, 1.4]},
    )

    reference_path = policy_evaluations[selected_names[0]].path_results[path_index]
    regimes = [step.weather_regime for step in reference_path.steps]
    max_year = max((step.ending_state.year for step in reference_path.steps), default=0)
    weather_axis.set_yticks([])
    weather_axis.set_ylabel("weather")
    weather_axis.set_xlim(-0.1, max(max_year, 1))
    for step_index, regime in enumerate(regimes):
        weather_axis.axvspan(step_index, step_index + 1, color="#d9e2f3", alpha=0.45)
        weather_axis.text(step_index + 0.5, 0.5, regime, ha="center", va="center", fontsize=8)
    weather_axis.set_frame_on(False)

    info_lines: list[str] = []
    for policy_name in selected_names:
        path = policy_evaluations[policy_name].path_results[path_index]
        steps = path.steps
        if not steps:
            continue
        x_values = [steps[0].starting_state.year]
        y_values = [steps[0].starting_state.cumulative_profit]
        x_values.extend(step.ending_state.year for step in steps)
        y_values.extend(step.ending_state.cumulative_profit for step in steps)

        profit_axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            label=policy_name,
        )
        if not steps[-1].ending_state.alive:
            profit_axis.scatter(
                [x_values[-1]],
                [y_values[-1]],
                marker="x",
                s=80,
                linewidths=2.0,
                color="#d62728",
            )
        action_sequence = " -> ".join(_action_name(step) for step in steps)
        terminal_state = path.final_state
        info_lines.append(
            (
                f"{policy_name}: survival={path.survival_years}, alive={terminal_state.alive}, "
                f"terminal_cash={terminal_state.cash:,.0f}, terminal_profit={terminal_state.cumulative_profit:,.0f}\n"
                f"  actions: {action_sequence}"
            )
        )

    profit_axis.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    profit_axis.set_xlabel("simulation year")
    profit_axis.set_ylabel("cumulative profit")
    profit_axis.grid(alpha=0.25)
    profit_axis.legend(loc="upper left")

    info_axis.axis("off")
    info_axis.text(
        0.0,
        1.0,
        "\n\n".join(info_lines) if info_lines else "no policy steps available",
        va="top",
        fontsize=9,
        family="monospace",
    )

    title_lines = _build_title_lines(
        default_title=f"policy cumulative profit trace (path {path_index})",
        title=title,
        subtitle_lines=subtitle_lines,
    )
    fig.suptitle("\n".join(title_lines), fontsize=11, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output
