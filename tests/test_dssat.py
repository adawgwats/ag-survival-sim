from __future__ import annotations

import sys
from pathlib import Path

from ag_survival_sim import (
    Action,
    DSSATExecutableConfig,
    DSSATExecutableCropModel,
    DSSATSummaryParser,
    FarmState,
    TemplateDSSATRunFactory,
)
from ag_survival_sim.scenario import AnnualScenario


def build_scenario(weather_regime: str = "normal") -> AnnualScenario:
    return AnnualScenario(
        year_index=0,
        weather_regime=weather_regime,
        weather_yield_multiplier=1.0,
        market_price_multiplier=1.0,
        operating_cost_multiplier=1.0,
    )


def test_dssat_summary_parser_reads_standard_header_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "Summary.OUT"
    summary_path.write_text(
        "\n".join(
            [
                "*DSSAT Summary Output",
                "@RUNNO TRNO CROP HWAM",
                " 1 1 CORN 8450",
                " 2 1 SOY 3210",
            ]
        ),
        encoding="utf-8",
    )

    parser = DSSATSummaryParser()
    records = parser.parse(summary_path)

    assert len(records) == 2
    assert records[0].get("HWAM") == 8450
    assert parser.select_record(records, {"RUNNO": 2}).get("CROP") == "SOY"


def test_dssat_summary_parser_handles_fixed_width_realistic_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "Summary.OUT"
    summary_path.write_text(
        "\n".join(
            [
                "*SUMMARY : real DSSAT output",
                "!IDENTIFIERS...",
                "@   RUNNO   TRNO CR EXNAME.. TNAM..................... FNAM.... WSTA.... HYEAR   HWAM",
                "        1      1 MZ IUAF9901 N=56 KG/HA POP=4.7 PL/M2 IUAF0001 IUAF9901  1999   5140",
            ]
        ),
        encoding="utf-8",
    )

    parser = DSSATSummaryParser()
    records = parser.parse(summary_path)

    assert len(records) == 1
    assert records[0].get("EXNAME") == "IUAF9901"
    assert records[0].get("TNAM") == "N=56 KG/HA POP=4.7 PL/M2"
    assert records[0].get("WSTA") == "IUAF9901"
    assert records[0].get("HWAM") == 5140


def test_dssat_executable_crop_model_runs_external_command_and_parses_summary(
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "DSSBatch.v48").write_text("fake batch file", encoding="utf-8")

    fake_dssat = tmp_path / "fake_dssat.py"
    fake_dssat.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "",
                "yield_value = Path('yield.txt').read_text(encoding='utf-8').strip()",
                "Path('Summary.OUT').write_text(",
                "    '*DSSAT Summary Output\\n@RUNNO TRNO CROP HWAM\\n 1 1 CORN ' + yield_value + '\\n',",
                "    encoding='utf-8',",
                ")",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )

    def render_scenario(working_dir: Path, state: FarmState, action: Action, scenario: AnnualScenario) -> None:
        del state, action
        value = "9100" if scenario.weather_regime == "good" else "7600"
        (working_dir / "yield.txt").write_text(value, encoding="utf-8")

    factory = TemplateDSSATRunFactory(
        template_dir=template_dir,
        workspace_root=tmp_path / "runs",
        batch_file="DSSBatch.v48",
        scenario_renderer=render_scenario,
        selector={"RUNNO": 1, "TRNO": 1},
    )
    crop_model = DSSATExecutableCropModel(
        run_factory=factory,
        config=DSSATExecutableConfig(
            executable=(sys.executable, str(fake_dssat)),
        ),
    )

    result = crop_model.yield_per_acre(
        state=FarmState.initial(),
        action=Action("corn", "low"),
        scenario=build_scenario("good"),
    )

    assert result == 9100.0
