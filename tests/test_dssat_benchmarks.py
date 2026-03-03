from __future__ import annotations

from pathlib import Path

import pytest

from ag_survival_sim import (
    Action,
    FarmState,
    build_benchmark_crop_model,
    discover_dssat_crop_inventory,
    get_benchmark_definition,
)
from ag_survival_sim.scenario import AnnualScenario


def test_discover_dssat_crop_inventory_finds_x_experiments(tmp_path: Path) -> None:
    root = tmp_path / "DSSAT48"
    root.mkdir()
    (root / "DSCSM048.EXE").write_text("fake", encoding="utf-8")

    maize = root / "Maize"
    maize.mkdir()
    (maize / "IUAF9901.MZX").write_text("x", encoding="utf-8")
    (maize / "IUAF9901.MZA").write_text("a", encoding="utf-8")

    soybean = root / "Soybean"
    soybean.mkdir()
    (soybean / "UFGA8401.SBX").write_text("x", encoding="utf-8")

    inventory = discover_dssat_crop_inventory(dssat_root=root)

    assert [(item.crop_directory, item.experiment_count) for item in inventory] == [
        ("Maize", 1),
        ("Soybean", 1),
    ]


def test_benchmark_registry_includes_multi_crop_entries() -> None:
    assert get_benchmark_definition("iowa_maize").crop == "corn"
    assert get_benchmark_definition("georgia_soybean").crop == "soy"
    assert get_benchmark_definition("kansas_wheat").crop == "wheat"
    assert get_benchmark_definition("dtsp_rice").crop == "rice"
    assert get_benchmark_definition("georgia_peanut").crop == "peanut"
    assert get_benchmark_definition("uafd_sunflower").crop == "sunflower"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("benchmark_name", "action"),
    (
        ("iowa_maize", Action("corn", "medium")),
        ("georgia_soybean", Action("soy", "medium")),
        ("kansas_wheat", Action("wheat", "medium")),
        ("dtsp_rice", Action("rice", "medium")),
        ("georgia_peanut", Action("peanut", "medium")),
        ("uafd_sunflower", Action("sunflower", "medium")),
    ),
)
def test_real_benchmark_crop_models_run_across_weather_regimes(
    benchmark_name: str,
    action: Action,
    tmp_path: Path,
) -> None:
    try:
        model = build_benchmark_crop_model(
            benchmark_name,
            workspace_root=tmp_path / benchmark_name,
        )
    except FileNotFoundError:
        pytest.skip("real DSSAT installation not available")

    state = FarmState.initial()
    yields = {}
    for regime in ("good", "normal", "drought"):
        yields[regime] = model.yield_per_acre(
            state=state,
            action=action,
            scenario=AnnualScenario(
                year_index=0,
                weather_regime=regime,
                weather_yield_multiplier=1.0,
                market_price_multiplier=1.0,
                operating_cost_multiplier=1.0,
            ),
        )

    assert yields["normal"] > 0.0
    assert yields["good"] >= yields["drought"]
