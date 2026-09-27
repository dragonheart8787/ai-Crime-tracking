"""DEV profile end to end through the CLI: simulate -> write Parquet -> validate (hash + invariants)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fcip.cli import main
from fcip.common.schemas import TABLE_NAMES


@pytest.mark.slow
def test_dev_profile_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "dev_seed42"
    assert main(["simulate", "--profile", "dev", "--seed", "42", "--out", str(out)]) == 0
    sim = json.loads(capsys.readouterr().out)
    for name in TABLE_NAMES:
        assert (out / f"{name}.parquet").is_file()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["dataset_hash"] == sim["dataset_hash"]
    assert 2000 <= meta["row_counts"]["persons"] <= 3000
    assert meta["sim_end"] == 90 * 86400
    assert {"numpy", "pyarrow", "polars", "pydantic", "python"} <= set(meta["package_versions"])
    assert set(meta["file_sha256"]) == set(TABLE_NAMES)
    assert main(["validate", "--data", str(out)]) == 0
    val = json.loads(capsys.readouterr().out)
    assert val["hash_matches_metadata"] and len(val["checks_passed"]) == 10
    # every family appears in DEV (min_instances: 1), including the OOD-designated ones
    assert set(meta["activation"]) == {
        "fan_in",
        "fan_out",
        "pass_through",
        "burst",
        "dormant_activation",
        "multi_hop",
        "cycle",
        "structuring_like",
        "shared_infrastructure",
        "account_to_cash",
    }
    assert all(v["instances"] >= 1 for v in meta["activation"].values())
