from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .scenario import AnnualScenario
from .types import Action, FarmState


class DSSATExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DSSATRunSpec:
    working_dir: Path
    batch_file: str
    command_mode: str = "B"
    external_control_file: str | None = None
    summary_output_file: str = "Summary.OUT"
    selector: Mapping[str, object] = field(default_factory=dict)
    yield_column: str | None = None
    cache_key: str | None = None


class DSSATRunFactory(Protocol):
    def prepare_run(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> DSSATRunSpec:
        ...


def identity_yield_transform(value: float, *_args: object, **_kwargs: object) -> float:
    return value


@dataclass(frozen=True)
class DSSATExecutableConfig:
    executable: Sequence[str]
    timeout_seconds: int = 300
    default_yield_column: str = "HWAM"
    environment: Mapping[str, str] | None = None
    yield_transform: Callable[[float, "DSSATSummaryRecord", Action], float] = identity_yield_transform

    def __post_init__(self) -> None:
        if not self.executable:
            raise ValueError("executable must contain at least one command component.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


@dataclass(frozen=True)
class DSSATSummaryRecord:
    values: Mapping[str, object]

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.values.get(key.upper(), default)


class DSSATSummaryParser:
    def parse(self, summary_output_path: Path) -> list[DSSATSummaryRecord]:
        if not summary_output_path.exists():
            raise FileNotFoundError(f"DSSAT summary output not found: {summary_output_path}")

        headers: list[str] | None = None
        records: list[DSSATSummaryRecord] = []

        for raw_line in summary_output_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("*") or line.startswith("!"):
                continue
            if line.startswith("@"):
                headers = [token.upper() for token in line[1:].split()]
                continue
            if headers is None:
                continue
            tokens = line.split()
            if len(tokens) < len(headers):
                continue
            row = {
                header: _coerce_value(token)
                for header, token in zip(headers, tokens[: len(headers)])
            }
            records.append(DSSATSummaryRecord(row))

        if not records:
            raise DSSATExecutionError(
                f"no DSSAT summary records parsed from {summary_output_path}"
            )
        return records

    def select_record(
        self,
        records: list[DSSATSummaryRecord],
        selector: Mapping[str, object],
    ) -> DSSATSummaryRecord:
        if not selector:
            return records[0]

        selector_upper = {key.upper(): value for key, value in selector.items()}
        for record in records:
            if all(_selector_matches(record.get(key), value) for key, value in selector_upper.items()):
                return record
        raise DSSATExecutionError(f"no DSSAT summary row matched selector {dict(selector)}")


@dataclass
class TemplateDSSATRunFactory:
    template_dir: Path
    workspace_root: Path
    batch_file: str
    scenario_renderer: Callable[[Path, FarmState, Action, AnnualScenario], None] | None = None
    external_control_file: str | None = None
    summary_output_file: str = "Summary.OUT"
    selector: Mapping[str, object] = field(default_factory=dict)

    def prepare_run(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> DSSATRunSpec:
        self.template_dir = self.template_dir.resolve()
        self.workspace_root = self.workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        run_name = self._run_name(state, action, scenario)
        working_dir = self.workspace_root / run_name
        if working_dir.exists():
            shutil.rmtree(working_dir)
        shutil.copytree(self.template_dir, working_dir)

        if self.scenario_renderer is not None:
            self.scenario_renderer(working_dir, state, action, scenario)

        return DSSATRunSpec(
            working_dir=working_dir,
            batch_file=self.batch_file,
            external_control_file=self.external_control_file,
            summary_output_file=self.summary_output_file,
            selector=self.selector,
            cache_key=run_name,
        )

    @staticmethod
    def _run_name(state: FarmState, action: Action, scenario: AnnualScenario) -> str:
        parts = (
            f"y{state.year}",
            f"p{scenario.year_index}",
            action.crop,
            action.input_level,
            scenario.weather_regime,
            f"ym{scenario.weather_yield_multiplier:.3f}",
            f"pm{scenario.market_price_multiplier:.3f}",
            f"cm{scenario.operating_cost_multiplier:.3f}",
        )
        return "_".join(part.replace(".", "p") for part in parts)


class DSSATExecutableCropModel:
    def __init__(
        self,
        *,
        run_factory: DSSATRunFactory,
        config: DSSATExecutableConfig,
        parser: DSSATSummaryParser | None = None,
    ) -> None:
        self.run_factory = run_factory
        self.config = config
        self.parser = parser or DSSATSummaryParser()

    def yield_per_acre(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> float:
        spec = self.run_factory.prepare_run(state=state, action=action, scenario=scenario)
        self._run_dssat(spec)
        summary_path = spec.working_dir / spec.summary_output_file
        records = self.parser.parse(summary_path)
        record = self.parser.select_record(records, spec.selector)

        yield_column = (spec.yield_column or self.config.default_yield_column).upper()
        raw_yield = record.get(yield_column)
        if raw_yield is None:
            raise DSSATExecutionError(
                f"yield column '{yield_column}' not present in {summary_path.name}"
            )
        try:
            numeric_yield = float(raw_yield)
        except (TypeError, ValueError) as error:
            raise DSSATExecutionError(
                f"yield value '{raw_yield}' in column '{yield_column}' is not numeric"
            ) from error

        return float(self.config.yield_transform(numeric_yield, record, action))

    def _run_dssat(self, spec: DSSATRunSpec) -> None:
        command = list(self.config.executable) + [spec.command_mode, spec.batch_file]
        if spec.external_control_file is not None:
            command.append(spec.external_control_file)

        environment = dict(os.environ)
        if self.config.environment is not None:
            environment.update(self.config.environment)

        completed = subprocess.run(
            command,
            cwd=spec.working_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise DSSATExecutionError(
                "DSSAT execution failed with return code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )


def _coerce_value(token: str) -> object:
    try:
        if any(character in token for character in (".", "E", "e")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _selector_matches(record_value: object | None, expected_value: object) -> bool:
    if record_value == expected_value:
        return True
    if record_value is None:
        return False
    return str(record_value).strip().upper() == str(expected_value).strip().upper()
