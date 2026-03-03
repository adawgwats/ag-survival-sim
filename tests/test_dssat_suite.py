from __future__ import annotations

from pathlib import Path

import pytest

from ag_survival_sim import (
    format_dssat_example_results,
    list_dssat_experiments,
    resolve_dssat_root,
    run_dssat_example,
)


def test_format_dssat_example_results_includes_status_columns() -> None:
    table = format_dssat_example_results(
        [
            run_dssat_example_result_fixture(
                experiment="EXAMPLE.MZX",
                succeeded=True,
                summary_rows=4,
                mean_hwam=8123.0,
            ),
            run_dssat_example_result_fixture(
                experiment="BROKEN.MZX",
                succeeded=False,
                summary_rows=0,
                error_message="missing weather",
            ),
        ]
    )

    assert "experiment" in table
    assert "EXAMPLE.MZX" in table
    assert "ok" in table
    assert "missing weather" in table


def test_list_dssat_experiments_auto_detects_crop_suffixes(tmp_path: Path) -> None:
    root = tmp_path / "DSSAT48"
    root.mkdir()
    (root / "DSCSM048.EXE").write_text("fake", encoding="utf-8")

    wheat = root / "Wheat"
    wheat.mkdir()
    (wheat / "RORO7401.WHX").write_text("x", encoding="utf-8")
    (wheat / "README.txt").write_text("ignore", encoding="utf-8")

    soybean = root / "Soybean"
    soybean.mkdir()
    (soybean / "UFGA8401.SBX").write_text("x", encoding="utf-8")

    assert list_dssat_experiments(root=root, crop_directory="Wheat") == ["RORO7401.WHX"]
    assert list_dssat_experiments(root=root, crop_directory="Soybean") == ["UFGA8401.SBX"]


@pytest.mark.integration
def test_real_dssat_example_runs_when_install_is_available(tmp_path: Path) -> None:
    try:
        root = resolve_dssat_root()
    except FileNotFoundError:
        pytest.skip("real DSSAT installation not available")

    experiments = list_dssat_experiments(root=root)
    assert "IUAF9901.MZX" in experiments

    result = run_dssat_example(
        experiment="IUAF9901.MZX",
        root=root,
        archive_root=tmp_path,
    )

    assert result.succeeded
    assert result.summary_rows == 4
    assert result.mean_hwam is not None
    assert result.mean_hwam > 5000
    assert len(result.treatment_names) == 4
    assert (tmp_path / "IUAF9901" / "Summary.OUT").exists()


def run_dssat_example_result_fixture(
    *,
    experiment: str,
    succeeded: bool,
    summary_rows: int,
    mean_hwam: float | None = None,
    error_message: str | None = None,
):
    from ag_survival_sim.dssat_suite import DSSATExampleResult

    return DSSATExampleResult(
        experiment=experiment,
        crop_directory="Maize",
        succeeded=succeeded,
        exit_code=0 if succeeded else 1,
        summary_rows=summary_rows,
        mean_hwam=mean_hwam,
        min_hwam=mean_hwam,
        max_hwam=mean_hwam,
        treatment_names=(),
        warning_tail="",
        error_message=error_message,
    )
