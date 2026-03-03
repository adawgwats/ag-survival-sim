from __future__ import annotations

from pathlib import Path
import csv

import pytest

from ag_survival_sim import (
    DSSATAllCropsSweepSummary,
    DSSATCropSweepSummary,
    format_dssat_example_results,
    format_dssat_all_crops_summary,
    export_dssat_treatment_rows,
    list_dssat_experiments,
    resolve_dssat_root,
    run_dssat_all_crops_sweep,
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


def test_run_dssat_all_crops_sweep_ignores_non_crop_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "DSSAT48"
    root.mkdir()
    (root / "DSCSM048.EXE").write_text("fake", encoding="utf-8")
    (root / "Maize").mkdir()
    (root / "Maize" / "IUAF9901.MZX").write_text("x", encoding="utf-8")
    (root / "Sequence").mkdir()
    (root / "Sequence" / "CHWC0012.SQX").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ag_survival_sim.dssat_suite.run_dssat_example_suite",
        lambda **kwargs: [
            run_dssat_example_result_fixture(
                experiment="IUAF9901.MZX",
                succeeded=True,
                summary_rows=2,
                mean_hwam=8000.0,
            )
        ],
    )

    _results, summary = run_dssat_all_crops_sweep(root=root, output_csv=tmp_path / "rows.csv")

    assert summary.crop_count == 1
    assert summary.crop_summaries[0].crop_directory == "Maize"


def test_format_dssat_all_crops_summary_lists_crop_counts() -> None:
    summary = DSSATAllCropsSweepSummary(
        crop_count=2,
        experiment_count=3,
        succeeded_count=3,
        failed_count=0,
        treatment_row_count=12,
        crop_summaries=(
            DSSATCropSweepSummary(
                crop_directory="Maize",
                experiment_count=2,
                succeeded_count=2,
                failed_count=0,
                treatment_row_count=8,
                mean_hwam=8123.0,
            ),
            DSSATCropSweepSummary(
                crop_directory="Soybean",
                experiment_count=1,
                succeeded_count=1,
                failed_count=0,
                treatment_row_count=4,
                mean_hwam=3241.0,
            ),
        ),
    )

    rendered = format_dssat_all_crops_summary(summary)

    assert "crop directories: 2" in rendered
    assert "Maize" in rendered
    assert "Soybean" in rendered


def test_export_dssat_treatment_rows_writes_csv(tmp_path: Path) -> None:
    result = run_dssat_example_result_fixture(
        experiment="EXAMPLE.MZX",
        succeeded=True,
        summary_rows=2,
        mean_hwam=8123.0,
    )
    output_csv = tmp_path / "rows.csv"

    export_dssat_treatment_rows([result], output_csv)

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["experiment"] == "EXAMPLE.MZX"
    assert rows[0]["crop_directory"] == "Maize"


def test_run_dssat_all_crops_sweep_aggregates_across_selected_crops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "ag_survival_sim.dssat_suite._list_discovered_crop_directories",
        lambda **_kwargs: ["Maize", "Soybean"],
    )

    fixtures = {
        "Maize": [
            run_dssat_example_result_fixture(
                experiment="A.MZX",
                succeeded=True,
                summary_rows=2,
                mean_hwam=8000.0,
            )
        ],
        "Soybean": [
            run_dssat_example_result_fixture(
                experiment="B.SBX",
                succeeded=False,
                summary_rows=0,
                error_message="boom",
            )
        ],
    }

    monkeypatch.setattr(
        "ag_survival_sim.dssat_suite.run_dssat_example_suite",
        lambda **kwargs: fixtures[kwargs["crop_directory"]],
    )

    output_csv = tmp_path / "unit_rows.csv"
    results, summary = run_dssat_all_crops_sweep(output_csv=output_csv)

    assert len(results) == 2
    assert summary.crop_count == 2
    assert summary.experiment_count == 2
    assert summary.succeeded_count == 1
    assert summary.failed_count == 1
    assert output_csv.exists()


def test_run_dssat_example_returns_failed_result_when_summary_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ag_survival_sim.dssat_suite import run_dssat_example

    root = tmp_path / "DSSAT48"
    root.mkdir()
    crop_path = root / "Maize"
    crop_path.mkdir()
    (root / "DSCSM048.EXE").write_text("fake", encoding="utf-8")
    (crop_path / "IUAF9901.MZX").write_text("x", encoding="utf-8")

    class _BrokenParser:
        def parse(self, _path: Path):
            raise FileNotFoundError("summary missing")

    monkeypatch.setattr(
        "ag_survival_sim.dssat_suite.subprocess.run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
    )

    result = run_dssat_example(
        experiment="IUAF9901.MZX",
        root=root,
        parser=_BrokenParser(),
    )

    assert not result.succeeded
    assert result.error_message == "summary missing"


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
    from ag_survival_sim.dssat_suite import DSSATExampleResult, DSSATTreatmentRow

    treatment_rows = ()
    if succeeded:
        treatment_rows = (
            DSSATTreatmentRow(
                crop_directory="Maize",
                experiment=experiment,
                runno=1,
                trno=1,
                crop_code="MZ",
                model="MZCER048",
                exname="EXAMPLE",
                treatment_name="baseline",
                field_name="FIELD1",
                weather_station="TEST",
                weather_year=1999,
                soil_id="SOIL1",
                hwam=mean_hwam,
                hwah=mean_hwam,
                cwam=mean_hwam,
                bwah=0.0,
                prcp=100.0,
                etcp=80.0,
                tmaxa=30.0,
                tmina=18.0,
                srada=20.0,
                crst="1",
            ),
        )

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
        treatment_rows=treatment_rows,
        warning_tail="",
        error_message=error_message,
    )
