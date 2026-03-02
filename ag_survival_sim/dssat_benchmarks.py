from __future__ import annotations

from pathlib import Path

from .dssat import DSSATExecutableConfig, DSSATExecutableCropModel
from .dssat_scenarios import (
    InstalledDSSATExperimentTemplate,
    InstalledDSSATRunFactory,
    dssat_hwam_to_action_units,
)
from .dssat_suite import resolve_dssat_root
from .simulator import FarmSimulator
from .types import Action


IOWA_MAIZE_ACTIONS: tuple[Action, ...] = (
    Action("corn", "low"),
    Action("corn", "medium"),
)


def build_iowa_maize_crop_model(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/iowa_maize",
) -> DSSATExecutableCropModel:
    root = resolve_dssat_root(dssat_root)
    template = InstalledDSSATExperimentTemplate(
        crop_directory="Maize",
        experiment_file="IUAF9901.MZX",
        action_treatment_map={
            ("corn", "low"): 1,
            ("corn", "medium"): 2,
        },
    )
    run_factory = InstalledDSSATRunFactory(
        template=template,
        workspace_root=Path(workspace_root),
        dssat_root=root,
    )
    config = DSSATExecutableConfig(
        executable=(str(root / "DSCSM048.EXE"),),
        yield_transform=dssat_hwam_to_action_units,
    )
    return DSSATExecutableCropModel(
        run_factory=run_factory,
        config=config,
    )


def build_iowa_maize_simulator(
    *,
    dssat_root: str | Path | None = None,
    workspace_root: str | Path = "dssat_runs/iowa_maize",
) -> FarmSimulator:
    return FarmSimulator(
        crop_model=build_iowa_maize_crop_model(
            dssat_root=dssat_root,
            workspace_root=workspace_root,
        )
    )
