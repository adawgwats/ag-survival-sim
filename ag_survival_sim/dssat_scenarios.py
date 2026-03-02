from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .dssat import DSSATRunSpec
from .dssat_suite import resolve_dssat_root
from .scenario import AnnualScenario
from .types import Action, FarmState


POUNDS_PER_KILOGRAM = 2.2046226218
ACRES_PER_HECTARE = 2.4710538147
POUNDS_PER_BUSHEL_BY_CROP: dict[str, float] = {
    "corn": 56.0,
    "soy": 60.0,
    "wheat": 60.0,
}


@dataclass(frozen=True)
class WeatherTransform:
    rain_multiplier: float = 1.0
    srad_multiplier: float = 1.0
    tmax_delta_c: float = 0.0
    tmin_delta_c: float = 0.0
    co2_delta_ppm: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.rain_multiplier == 1.0
            and self.srad_multiplier == 1.0
            and self.tmax_delta_c == 0.0
            and self.tmin_delta_c == 0.0
            and self.co2_delta_ppm == 0.0
        )


DEFAULT_TRANSFORM_BY_REGIME: dict[str, WeatherTransform] = {
    "good": WeatherTransform(
        rain_multiplier=1.10,
        srad_multiplier=1.04,
        tmax_delta_c=0.4,
        tmin_delta_c=0.2,
    ),
    "normal": WeatherTransform(),
    "drought": WeatherTransform(
        rain_multiplier=0.45,
        srad_multiplier=1.05,
        tmax_delta_c=2.8,
        tmin_delta_c=1.4,
    ),
}


@dataclass(frozen=True)
class InstalledDSSATExperimentTemplate:
    crop_directory: str
    experiment_file: str
    action_treatment_map: Mapping[tuple[str, str], int]
    weather_transforms: Mapping[str, WeatherTransform] = field(
        default_factory=lambda: DEFAULT_TRANSFORM_BY_REGIME
    )
    summary_output_file: str = "Summary.OUT"
    yield_column: str = "HWAM"


@dataclass
class InstalledDSSATRunFactory:
    template: InstalledDSSATExperimentTemplate
    workspace_root: Path
    dssat_root: Path | None = None

    def prepare_run(
        self,
        *,
        state: FarmState,
        action: Action,
        scenario: AnnualScenario,
    ) -> DSSATRunSpec:
        del state
        dssat_root = resolve_dssat_root(self.dssat_root)
        crop_dir = dssat_root / self.template.crop_directory
        experiment_path = crop_dir / self.template.experiment_file
        if not experiment_path.exists():
            raise FileNotFoundError(f"DSSAT experiment template not found: {experiment_path}")

        treatment_number = self.template.action_treatment_map.get(action.key)
        if treatment_number is None:
            raise KeyError(f"no DSSAT treatment mapping for action {action.key}")

        weather_code = read_weather_code(experiment_path)
        weather_source = dssat_root / "Weather" / f"{weather_code}.WTH"
        if not weather_source.exists():
            raise FileNotFoundError(f"DSSAT weather file not found: {weather_source}")

        run_name = build_run_name(
            experiment_stem=experiment_path.stem,
            action=action,
            scenario=scenario,
        )
        working_dir = self.workspace_root.resolve() / run_name
        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        run_experiment = working_dir / experiment_path.name
        run_weather = working_dir / weather_source.name
        shutil.copy2(experiment_path, run_experiment)
        shutil.copy2(weather_source, run_weather)

        transform = self.template.weather_transforms.get(
            scenario.weather_regime,
            WeatherTransform(),
        )
        if not transform.is_identity:
            apply_weather_transform(run_weather, run_weather, transform)

        return DSSATRunSpec(
            working_dir=working_dir,
            batch_file=run_experiment.name,
            command_mode="A",
            summary_output_file=self.template.summary_output_file,
            yield_column=self.template.yield_column,
            selector={"TRNO": treatment_number},
            cache_key=run_name,
        )


def build_run_name(
    *,
    experiment_stem: str,
    action: Action,
    scenario: AnnualScenario,
) -> str:
    return "_".join(
        [
            experiment_stem,
            action.crop,
            action.input_level,
            scenario.weather_regime,
            f"y{scenario.year_index}",
            f"ym{scenario.weather_yield_multiplier:.3f}".replace(".", "p"),
            f"pm{scenario.market_price_multiplier:.3f}".replace(".", "p"),
        ]
    )


def read_weather_code(experiment_path: Path) -> str:
    lines = experiment_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_fields = False
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("*FIELDS"):
            in_fields = True
            continue
        if in_fields and line.startswith("*"):
            break
        if not in_fields or not line.strip() or line.lstrip().startswith("@"):
            continue
        tokens = line.split()
        if len(tokens) >= 3:
            return tokens[2].strip()
    raise ValueError(f"could not parse weather code from {experiment_path}")


def apply_weather_transform(
    source_path: Path,
    destination_path: Path,
    transform: WeatherTransform,
) -> None:
    output_lines: list[str] = []
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_daily_data = False
    in_metadata = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            output_lines.append(raw_line)
            continue
        if raw_line.startswith("@ INSI"):
            in_metadata = True
            in_daily_data = False
            output_lines.append(raw_line)
            continue
        if raw_line.startswith("@DATE"):
            in_metadata = False
            in_daily_data = True
            output_lines.append(raw_line)
            continue
        if raw_line.startswith("*") or raw_line.startswith("@"):
            in_metadata = False
            in_daily_data = False
            output_lines.append(raw_line)
            continue

        if in_metadata:
            output_lines.append(_transform_metadata_line(raw_line, transform))
            in_metadata = False
            continue
        if in_daily_data:
            output_lines.append(_transform_weather_data_line(raw_line, transform))
            continue
        output_lines.append(raw_line)

    destination_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _transform_metadata_line(line: str, transform: WeatherTransform) -> str:
    if transform.co2_delta_ppm == 0.0:
        return line
    tokens = line.split()
    if len(tokens) < 9:
        return line
    try:
        co2 = float(tokens[-1])
    except ValueError:
        return line
    updated = co2 + transform.co2_delta_ppm
    tokens[-1] = f"{updated:.0f}"
    return (
        f"{tokens[0]:>6} {float(tokens[1]):>8.3f} {float(tokens[2]):>9.3f} "
        f"{float(tokens[3]):>6.0f} {float(tokens[4]):>5.1f} {float(tokens[5]):>5.1f} "
        f"{float(tokens[6]):>5.1f} {float(tokens[7]):>6.2f} {float(tokens[8]):>4.0f}"
    )


def _transform_weather_data_line(line: str, transform: WeatherTransform) -> str:
    tokens = line.split()
    if len(tokens) < 5:
        return line
    date = tokens[0]
    srad = _safe_float(tokens[1])
    tmax = _safe_float(tokens[2])
    tmin = _safe_float(tokens[3])
    rain = _safe_float(tokens[4])

    if srad is not None and srad > -90.0:
        srad = max(srad * transform.srad_multiplier, 0.0)
    if tmax is not None and tmax > -90.0:
        tmax = tmax + transform.tmax_delta_c
    if tmin is not None and tmin > -90.0:
        tmin = tmin + transform.tmin_delta_c
    if rain is not None and rain > -90.0:
        rain = max(rain * transform.rain_multiplier, 0.0)

    return (
        f"{date:>5}"
        f"{_format_weather_value(srad, width=6)}"
        f"{_format_weather_value(tmax, width=6)}"
        f"{_format_weather_value(tmin, width=6)}"
        f"{_format_weather_value(rain, width=6)}"
    )


def kg_per_hectare_to_bushels_per_acre(
    yield_kg_per_hectare: float,
    *,
    crop: str,
) -> float:
    pounds_per_bushel = POUNDS_PER_BUSHEL_BY_CROP.get(crop)
    if pounds_per_bushel is None:
        raise KeyError(f"no bushel conversion configured for crop '{crop}'")
    pounds_per_hectare = yield_kg_per_hectare * POUNDS_PER_KILOGRAM
    bushels_per_hectare = pounds_per_hectare / pounds_per_bushel
    return bushels_per_hectare / ACRES_PER_HECTARE


def dssat_hwam_to_action_units(
    value: float,
    _record: object,
    action: Action,
) -> float:
    return kg_per_hectare_to_bushels_per_acre(value, crop=action.crop)


def _safe_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _format_weather_value(value: float | None, *, width: int) -> str:
    if value is None:
        return f"{-99.0:>{width}.1f}"
    return f"{value:>{width}.1f}"
