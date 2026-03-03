from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .dssat import DSSATExecutableConfig, DSSATExecutableCropModel
from .dssat_scenarios import (
    InstalledDSSATExperimentTemplate,
    InstalledDSSATRunFactory,
    dssat_hwam_to_action_units,
    kg_per_hectare_to_hundredweight_per_acre,
    kg_per_hectare_to_short_tons_per_acre,
)
from .dssat_suite import NON_CROP_EXPERIMENT_DIRECTORIES, resolve_dssat_root
from .simulator import FarmSimulator
from .types import Action


YieldTransform = Callable[[float, object, Action], float]


@dataclass(frozen=True)
class DSSATCropInventory:
    crop_directory: str
    experiment_count: int
    experiment_suffixes: tuple[str, ...]
    sample_experiment: str | None


@dataclass(frozen=True)
class DSSATBenchmarkDefinition:
    name: str
    crop: str
    crop_directory: str
    experiment_file: str
    actions: tuple[Action, ...]
    action_treatment_map: Mapping[tuple[str, str], int]
    yield_transform: YieldTransform = dssat_hwam_to_action_units
    yield_column: str = "HWAM"
    description: str = ""


BENCHMARK_DEFINITIONS: dict[str, DSSATBenchmarkDefinition] = {
    "iowa_maize": DSSATBenchmarkDefinition(
        name="iowa_maize",
        crop="corn",
        crop_directory="Maize",
        experiment_file="IUAF9901.MZX",
        actions=(
            Action("corn", "low"),
            Action("corn", "medium"),
        ),
        action_treatment_map={
            ("corn", "low"): 1,
            ("corn", "medium"): 2,
        },
        description="Iowa maize nitrogen response benchmark built from IUAF9901.",
    ),
    "georgia_maize_management": DSSATBenchmarkDefinition(
        name="georgia_maize_management",
        crop="corn",
        crop_directory="Maize",
        experiment_file="UFGA8201.MZX",
        actions=(
            Action("corn", "rainfed_low"),
            Action("corn", "rainfed_high"),
            Action("corn", "irrigated_low"),
            Action("corn", "irrigated_high"),
        ),
        action_treatment_map={
            ("corn", "rainfed_low"): 1,
            ("corn", "rainfed_high"): 2,
            ("corn", "irrigated_low"): 3,
            ("corn", "irrigated_high"): 4,
        },
        description="Georgia maize irrigation and nitrogen benchmark built from UFGA8201.",
    ),
    "georgia_soybean": DSSATBenchmarkDefinition(
        name="georgia_soybean",
        crop="soy",
        crop_directory="Soybean",
        experiment_file="UFGA8401.SBX",
        actions=(
            Action("soy", "low"),
            Action("soy", "medium"),
        ),
        action_treatment_map={
            ("soy", "low"): 2,
            ("soy", "medium"): 1,
        },
        description="Georgia soybean irrigated vs. rainfed benchmark built from UFGA8401.",
    ),
    "kansas_wheat": DSSATBenchmarkDefinition(
        name="kansas_wheat",
        crop="wheat",
        crop_directory="Wheat",
        experiment_file="KSAS8101.WHX",
        actions=(
            Action("wheat", "low"),
            Action("wheat", "medium"),
        ),
        action_treatment_map={
            ("wheat", "low"): 1,
            ("wheat", "medium"): 2,
        },
        description="Kansas wheat nitrogen response benchmark built from KSAS8101.",
    ),
    "dtsp_rice": DSSATBenchmarkDefinition(
        name="dtsp_rice",
        crop="rice",
        crop_directory="Rice",
        experiment_file="DTSP8502.RIX",
        actions=(
            Action("rice", "low"),
            Action("rice", "medium"),
        ),
        action_treatment_map={
            ("rice", "low"): 2,
            ("rice", "medium"): 4,
        },
        yield_transform=lambda value, _record, _action: kg_per_hectare_to_hundredweight_per_acre(value),
        description="Rice nitrogen response benchmark built from DTSP8502.",
    ),
    "georgia_peanut": DSSATBenchmarkDefinition(
        name="georgia_peanut",
        crop="peanut",
        crop_directory="Peanut",
        experiment_file="UFGA8701.PNX",
        actions=(
            Action("peanut", "low"),
            Action("peanut", "medium"),
        ),
        action_treatment_map={
            ("peanut", "low"): 2,
            ("peanut", "medium"): 1,
        },
        yield_transform=lambda value, _record, _action: kg_per_hectare_to_short_tons_per_acre(value),
        description="Georgia peanut disease-control benchmark built from UFGA8701.",
    ),
    "uafd_sunflower": DSSATBenchmarkDefinition(
        name="uafd_sunflower",
        crop="sunflower",
        crop_directory="Sunflower",
        experiment_file="UAFD0801.SUX",
        actions=(
            Action("sunflower", "low"),
            Action("sunflower", "medium"),
        ),
        action_treatment_map={
            ("sunflower", "low"): 2,
            ("sunflower", "medium"): 4,
        },
        yield_transform=lambda value, _record, _action: kg_per_hectare_to_hundredweight_per_acre(value),
        description="Sunflower nitrogen response benchmark built from UAFD0801.",
    ),
}


IOWA_MAIZE_ACTIONS = BENCHMARK_DEFINITIONS["iowa_maize"].actions
GEORGIA_MAIZE_MANAGEMENT_ACTIONS = BENCHMARK_DEFINITIONS["georgia_maize_management"].actions
GEORGIA_SOYBEAN_ACTIONS = BENCHMARK_DEFINITIONS["georgia_soybean"].actions
KANSAS_WHEAT_ACTIONS = BENCHMARK_DEFINITIONS["kansas_wheat"].actions
DTSP_RICE_ACTIONS = BENCHMARK_DEFINITIONS["dtsp_rice"].actions
GEORGIA_PEANUT_ACTIONS = BENCHMARK_DEFINITIONS["georgia_peanut"].actions
UAFD_SUNFLOWER_ACTIONS = BENCHMARK_DEFINITIONS["uafd_sunflower"].actions


def discover_dssat_crop_inventory(
    *,
    dssat_root: str | Path | None = None,
) -> list[DSSATCropInventory]:
    root = resolve_dssat_root(dssat_root)
    inventories: list[DSSATCropInventory] = []

    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if directory.name in NON_CROP_EXPERIMENT_DIRECTORIES:
            continue
        experiment_files = sorted(_iter_experiment_files(directory))
        if not experiment_files:
            continue
        suffixes = tuple(sorted({path.suffix.upper() for path in experiment_files}))
        inventories.append(
            DSSATCropInventory(
                crop_directory=directory.name,
                experiment_count=len(experiment_files),
                experiment_suffixes=suffixes,
                sample_experiment=experiment_files[0].name,
            )
        )

    return inventories


def list_benchmark_definitions() -> tuple[DSSATBenchmarkDefinition, ...]:
    return tuple(BENCHMARK_DEFINITIONS[name] for name in sorted(BENCHMARK_DEFINITIONS))


def get_benchmark_definition(name: str) -> DSSATBenchmarkDefinition:
    try:
        return BENCHMARK_DEFINITIONS[name]
    except KeyError as error:
        raise KeyError(f"unknown DSSAT benchmark '{name}'") from error


def build_benchmark_crop_model(
    benchmark_name: str,
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/benchmark",
) -> DSSATExecutableCropModel:
    benchmark = get_benchmark_definition(benchmark_name)
    root = resolve_dssat_root(dssat_root)
    template = InstalledDSSATExperimentTemplate(
        crop_directory=benchmark.crop_directory,
        experiment_file=benchmark.experiment_file,
        action_treatment_map=benchmark.action_treatment_map,
        yield_column=benchmark.yield_column,
    )
    run_factory = InstalledDSSATRunFactory(
        template=template,
        workspace_root=Path(workspace_root),
        dssat_root=root,
    )
    config = DSSATExecutableConfig(
        executable=(str(root / "DSCSM048.EXE"),),
        yield_transform=benchmark.yield_transform,
    )
    return DSSATExecutableCropModel(
        run_factory=run_factory,
        config=config,
    )


def build_benchmark_simulator(
    benchmark_name: str,
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/benchmark",
) -> FarmSimulator:
    return FarmSimulator(
        crop_model=build_benchmark_crop_model(
            benchmark_name,
            dssat_root=dssat_root,
            workspace_root=workspace_root,
        )
    )


def build_iowa_maize_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/iowa_maize",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "iowa_maize",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_iowa_maize_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/iowa_maize",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "iowa_maize",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_soybean_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_soybean",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "georgia_soybean",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_soybean_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_soybean",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "georgia_soybean",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_maize_management_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_maize_management",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "georgia_maize_management",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_maize_management_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_maize_management",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "georgia_maize_management",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_kansas_wheat_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/kansas_wheat",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "kansas_wheat",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_kansas_wheat_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/kansas_wheat",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "kansas_wheat",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_dtsp_rice_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/dtsp_rice",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "dtsp_rice",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_dtsp_rice_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/dtsp_rice",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "dtsp_rice",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_peanut_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_peanut",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "georgia_peanut",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_georgia_peanut_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/georgia_peanut",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "georgia_peanut",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_uafd_sunflower_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/uafd_sunflower",
) -> DSSATExecutableCropModel:
    return build_benchmark_crop_model(
        "uafd_sunflower",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def build_uafd_sunflower_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/uafd_sunflower",
) -> FarmSimulator:
    return build_benchmark_simulator(
        "uafd_sunflower",
        dssat_root=dssat_root,
        workspace_root=workspace_root,
    )


def format_crop_inventory(inventory: list[DSSATCropInventory]) -> str:
    if not inventory:
        return "No DSSAT crops with experiment templates were discovered."

    headers = ("crop_directory", "experiments", "suffixes", "sample_experiment")
    rows = [headers]
    for item in inventory:
        rows.append(
            (
                item.crop_directory,
                str(item.experiment_count),
                ",".join(item.experiment_suffixes),
                item.sample_experiment or "-",
            )
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    formatted: list[str] = []
    for index, row in enumerate(rows):
        formatted.append("  ".join(value.ljust(widths[column]) for column, value in enumerate(row)))
        if index == 0:
            formatted.append("  ".join("-" * width for width in widths))
    return "\n".join(formatted)


def _iter_experiment_files(directory: Path):
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.upper()
        if len(suffix) == 4 and suffix.endswith("X") and suffix[1:].isalnum():
            yield path
