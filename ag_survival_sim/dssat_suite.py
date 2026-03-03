from __future__ import annotations

import os
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .dssat import DSSATExecutionError, DSSATSummaryParser

NON_CROP_EXPERIMENT_DIRECTORIES = {
    "ClimateChange",
    "Seasonal",
    "Sequence",
    "Spatial",
    "YieldForecast",
}


@dataclass(frozen=True)
class DSSATExampleResult:
    experiment: str
    crop_directory: str
    succeeded: bool
    exit_code: int
    summary_rows: int
    mean_hwam: float | None
    min_hwam: float | None
    max_hwam: float | None
    treatment_names: tuple[str, ...]
    treatment_rows: tuple["DSSATTreatmentRow", ...]
    warning_tail: str
    error_message: str | None = None


@dataclass(frozen=True)
class DSSATTreatmentRow:
    crop_directory: str
    experiment: str
    runno: int | None
    trno: int | None
    crop_code: str
    model: str
    exname: str
    treatment_name: str
    field_name: str
    weather_station: str
    weather_year: int | None
    soil_id: str
    hwam: float | None
    hwah: float | None
    cwam: float | None
    bwah: float | None
    prcp: float | None
    etcp: float | None
    tmaxa: float | None
    tmina: float | None
    srada: float | None
    crst: str


@dataclass(frozen=True)
class DSSATCropSweepSummary:
    crop_directory: str
    experiment_count: int
    succeeded_count: int
    failed_count: int
    treatment_row_count: int
    mean_hwam: float | None


@dataclass(frozen=True)
class DSSATAllCropsSweepSummary:
    crop_count: int
    experiment_count: int
    succeeded_count: int
    failed_count: int
    treatment_row_count: int
    crop_summaries: tuple[DSSATCropSweepSummary, ...]


def resolve_dssat_root(root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root))
    env_root = os.environ.get("DSSAT_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(r"C:\DSSAT48"))

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "DSCSM048.EXE").exists():
            return resolved
    raise FileNotFoundError(
        "Could not find a DSSAT installation. Set DSSAT_ROOT or install DSSAT at C:\\DSSAT48."
    )


def list_dssat_experiments(
    *,
    root: str | Path | None = None,
    crop_directory: str = "Maize",
    suffix: str | None = None,
) -> list[str]:
    dssat_root = resolve_dssat_root(root)
    crop_path = dssat_root / crop_directory
    if not crop_path.exists():
        raise FileNotFoundError(f"DSSAT crop directory not found: {crop_path}")
    if suffix is not None:
        return sorted(path.name for path in crop_path.glob(f"*{suffix}"))
    return sorted(path.name for path in _iter_experiment_files(crop_path))


def run_dssat_example(
    *,
    experiment: str,
    root: str | Path | None = None,
    crop_directory: str = "Maize",
    parser: DSSATSummaryParser | None = None,
    archive_root: str | Path | None = None,
) -> DSSATExampleResult:
    dssat_root = resolve_dssat_root(root)
    parser = parser or DSSATSummaryParser()
    crop_path = dssat_root / crop_directory
    experiment_path = crop_path / experiment
    executable = dssat_root / "DSCSM048.EXE"

    if not experiment_path.exists():
        raise FileNotFoundError(f"DSSAT experiment not found: {experiment_path}")

    _clear_dssat_outputs(crop_path)

    completed = subprocess.run(
        [str(executable), "A", experiment],
        cwd=crop_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    warning_tail = _read_warning_tail(crop_path / "WARNING.OUT")

    if archive_root is not None:
        _archive_outputs(crop_path, Path(archive_root), experiment_path.stem)

    if completed.returncode != 0:
        return DSSATExampleResult(
            experiment=experiment,
            crop_directory=crop_directory,
            succeeded=False,
            exit_code=completed.returncode,
            summary_rows=0,
            mean_hwam=None,
            min_hwam=None,
            max_hwam=None,
            treatment_names=(),
            treatment_rows=(),
            warning_tail=warning_tail,
            error_message=(completed.stderr.strip() or completed.stdout.strip() or None),
        )

    summary_path = crop_path / "Summary.OUT"
    try:
        records = parser.parse(summary_path)
    except (FileNotFoundError, DSSATExecutionError) as error:
        return DSSATExampleResult(
            experiment=experiment,
            crop_directory=crop_directory,
            succeeded=False,
            exit_code=completed.returncode,
            summary_rows=0,
            mean_hwam=None,
            min_hwam=None,
            max_hwam=None,
            treatment_names=(),
            treatment_rows=(),
            warning_tail=warning_tail,
            error_message=str(error),
        )
    treatment_rows = tuple(
        _build_treatment_row(
            crop_directory=crop_directory,
            experiment=experiment,
            record=record,
        )
        for record in records
    )
    hwam_values = [
        float(record.get("HWAM"))
        for record in records
        if isinstance(record.get("HWAM"), (int, float))
    ]
    treatment_names = tuple(
        str(record.get("TNAM")).strip()
        for record in records
        if str(record.get("TNAM", "")).strip()
    )

    return DSSATExampleResult(
        experiment=experiment,
        crop_directory=crop_directory,
        succeeded=True,
        exit_code=completed.returncode,
        summary_rows=len(records),
        mean_hwam=(sum(hwam_values) / len(hwam_values)) if hwam_values else None,
        min_hwam=min(hwam_values) if hwam_values else None,
        max_hwam=max(hwam_values) if hwam_values else None,
        treatment_names=treatment_names,
        treatment_rows=treatment_rows,
        warning_tail=warning_tail,
    )


def run_dssat_example_suite(
    *,
    root: str | Path | None = None,
    crop_directory: str = "Maize",
    experiments: Sequence[str] | None = None,
    archive_root: str | Path | None = None,
) -> list[DSSATExampleResult]:
    parser = DSSATSummaryParser()
    selected = list(experiments or list_dssat_experiments(root=root, crop_directory=crop_directory))
    return [
        run_dssat_example(
            experiment=experiment,
            root=root,
            crop_directory=crop_directory,
            parser=parser,
            archive_root=archive_root,
        )
        for experiment in selected
    ]


def format_dssat_example_results(results: Iterable[DSSATExampleResult]) -> str:
    rows = list(results)
    if not rows:
        return "No DSSAT example results."

    headers = ("experiment", "rows", "mean_hwam", "min_hwam", "max_hwam", "status")
    body: list[tuple[str, ...]] = [headers]

    for result in rows:
        status = "ok" if result.succeeded else f"failed: {result.error_message or 'unknown error'}"
        body.append(
            (
                result.experiment,
                str(result.summary_rows),
                _format_float(result.mean_hwam),
                _format_float(result.min_hwam),
                _format_float(result.max_hwam),
                status,
            )
        )

    widths = [max(len(row[index]) for row in body) for index in range(len(headers))]
    formatted_rows = []
    for index, row in enumerate(body):
        formatted = "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        formatted_rows.append(formatted)
        if index == 0:
            formatted_rows.append("  ".join("-" * width for width in widths))
    return "\n".join(formatted_rows)


def run_dssat_all_crops_sweep(
    *,
    root: str | Path | None = None,
    crop_directories: Sequence[str] | None = None,
    archive_root: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> tuple[list[DSSATExampleResult], DSSATAllCropsSweepSummary]:
    selected_crops = list(crop_directories or _list_discovered_crop_directories(root=root))
    all_results: list[DSSATExampleResult] = []
    crop_summaries: list[DSSATCropSweepSummary] = []

    for crop_directory in selected_crops:
        crop_archive_root = None
        if archive_root is not None:
            crop_archive_root = Path(archive_root) / crop_directory
        results = run_dssat_example_suite(
            root=root,
            crop_directory=crop_directory,
            archive_root=crop_archive_root,
        )
        all_results.extend(results)

        successful = [result for result in results if result.succeeded]
        hwam_values = [
            result.mean_hwam
            for result in successful
            if result.mean_hwam is not None
        ]
        crop_summaries.append(
            DSSATCropSweepSummary(
                crop_directory=crop_directory,
                experiment_count=len(results),
                succeeded_count=sum(1 for result in results if result.succeeded),
                failed_count=sum(1 for result in results if not result.succeeded),
                treatment_row_count=sum(len(result.treatment_rows) for result in results),
                mean_hwam=(sum(hwam_values) / len(hwam_values)) if hwam_values else None,
            )
        )

    if output_csv is not None:
        export_dssat_treatment_rows(all_results, output_csv)

    summary = DSSATAllCropsSweepSummary(
        crop_count=len(crop_summaries),
        experiment_count=len(all_results),
        succeeded_count=sum(summary.succeeded_count for summary in crop_summaries),
        failed_count=sum(summary.failed_count for summary in crop_summaries),
        treatment_row_count=sum(summary.treatment_row_count for summary in crop_summaries),
        crop_summaries=tuple(crop_summaries),
    )
    return all_results, summary


def format_dssat_all_crops_summary(summary: DSSATAllCropsSweepSummary) -> str:
    lines = [
        "DSSAT all-crops sweep summary",
        f"crop directories: {summary.crop_count}",
        f"experiments: {summary.experiment_count}",
        f"succeeded: {summary.succeeded_count}",
        f"failed: {summary.failed_count}",
        f"treatment rows: {summary.treatment_row_count}",
        "",
        "crop_directory    experiments  succeeded  failed  rows  mean_hwam",
        "----------------  -----------  ---------  ------  ----  ---------",
    ]
    for crop_summary in summary.crop_summaries:
        mean_hwam = _format_float(crop_summary.mean_hwam)
        lines.append(
            f"{crop_summary.crop_directory:<16}"
            f"  {crop_summary.experiment_count:>11}"
            f"  {crop_summary.succeeded_count:>9}"
            f"  {crop_summary.failed_count:>6}"
            f"  {crop_summary.treatment_row_count:>4}"
            f"  {mean_hwam:>9}"
        )
    return "\n".join(lines)


def export_dssat_treatment_rows(
    results: Sequence[DSSATExampleResult],
    output_csv: str | Path,
) -> None:
    rows = [row for result in results for row in result.treatment_rows]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_treatment_row_to_dict(rows[0]).keys()) if rows else _treatment_row_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(_treatment_row_to_dict(row))


def _clear_dssat_outputs(crop_path: Path) -> None:
    for name in ("Summary.OUT", "OVERVIEW.OUT", "PlantGro.OUT", "WARNING.OUT", "ERROR.OUT"):
        (crop_path / name).unlink(missing_ok=True)


def _read_warning_tail(path: Path) -> str:
    if not path.exists():
        return ""
    return " ".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-2:]).strip()


def _archive_outputs(crop_path: Path, archive_root: Path, experiment_stem: str) -> None:
    destination = archive_root / experiment_stem
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("Summary.OUT", "OVERVIEW.OUT", "PlantGro.OUT", "WARNING.OUT", "ERROR.OUT"):
        source = crop_path / name
        if source.exists():
            shutil.copy2(source, destination / name)


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _build_treatment_row(
    *,
    crop_directory: str,
    experiment: str,
    record: DSSATSummaryRecord,
) -> DSSATTreatmentRow:
    return DSSATTreatmentRow(
        crop_directory=crop_directory,
        experiment=experiment,
        runno=_coerce_int(record.get("RUNNO")),
        trno=_coerce_int(record.get("TRNO")),
        crop_code=_coerce_text(record.get("CR")),
        model=_coerce_text(record.get("MODEL")),
        exname=_coerce_text(record.get("EXNAME")),
        treatment_name=_coerce_text(record.get("TNAM")),
        field_name=_coerce_text(record.get("FNAM")),
        weather_station=_coerce_text(record.get("WSTA")),
        weather_year=_coerce_int(record.get("WYEAR")),
        soil_id=_coerce_text(record.get("SOIL_ID")),
        hwam=_coerce_float(record.get("HWAM")),
        hwah=_coerce_float(record.get("HWAH")),
        cwam=_coerce_float(record.get("CWAM")),
        bwah=_coerce_float(record.get("BWAH")),
        prcp=_coerce_float(record.get("PRCP")),
        etcp=_coerce_float(record.get("ETCP")),
        tmaxa=_coerce_float(record.get("TMAXA")),
        tmina=_coerce_float(record.get("TMINA")),
        srada=_coerce_float(record.get("SRADA")),
        crst=_coerce_text(record.get("CRST")),
    )


def _coerce_int(value: object | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _coerce_float(value: object | None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _list_discovered_crop_directories(*, root: str | Path | None = None) -> list[str]:
    dssat_root = resolve_dssat_root(root)
    discovered: list[str] = []
    for path in sorted(candidate for candidate in dssat_root.iterdir() if candidate.is_dir() and not candidate.name.startswith(".")):
        if path.name in NON_CROP_EXPERIMENT_DIRECTORIES:
            continue
        if any(_iter_experiment_files(path)):
            discovered.append(path.name)
    return discovered


def _treatment_row_fieldnames() -> list[str]:
    return [
        "crop_directory",
        "experiment",
        "runno",
        "trno",
        "crop_code",
        "model",
        "exname",
        "treatment_name",
        "field_name",
        "weather_station",
        "weather_year",
        "soil_id",
        "hwam",
        "hwah",
        "cwam",
        "bwah",
        "prcp",
        "etcp",
        "tmaxa",
        "tmina",
        "srada",
        "crst",
    ]


def _treatment_row_to_dict(row: DSSATTreatmentRow) -> dict[str, object | None]:
    return {
        "crop_directory": row.crop_directory,
        "experiment": row.experiment,
        "runno": row.runno,
        "trno": row.trno,
        "crop_code": row.crop_code,
        "model": row.model,
        "exname": row.exname,
        "treatment_name": row.treatment_name,
        "field_name": row.field_name,
        "weather_station": row.weather_station,
        "weather_year": row.weather_year,
        "soil_id": row.soil_id,
        "hwam": row.hwam,
        "hwah": row.hwah,
        "cwam": row.cwam,
        "bwah": row.bwah,
        "prcp": row.prcp,
        "etcp": row.etcp,
        "tmaxa": row.tmaxa,
        "tmina": row.tmina,
        "srada": row.srada,
        "crst": row.crst,
    }


def _iter_experiment_files(crop_path: Path):
    for path in crop_path.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.upper()
        if len(suffix) == 4 and suffix.endswith("X") and suffix[1:].isalnum():
            yield path
