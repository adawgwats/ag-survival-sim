from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .dssat import DSSATSummaryParser


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
    warning_tail: str
    error_message: str | None = None


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
            warning_tail=warning_tail,
            error_message=(completed.stderr.strip() or completed.stdout.strip() or None),
        )

    summary_path = crop_path / "Summary.OUT"
    records = parser.parse(summary_path)
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


def _iter_experiment_files(crop_path: Path):
    for path in crop_path.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.upper()
        if len(suffix) == 4 and suffix.endswith("X") and suffix[1:].isalnum():
            yield path
