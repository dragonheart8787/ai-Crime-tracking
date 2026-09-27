"""Same seed -> identical dataset hash (in-process and across fresh processes with different PYTHONHASHSEED);
different seeds -> different data."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from fcip.config.loader import load_config
from fcip.simulation.generator import generate

_SNIPPET = (
    "import json; from fcip.config.loader import load_config; "
    "from fcip.simulation.generator import generate; "
    "m = generate(load_config('tiny')).metadata; "
    "print(json.dumps({'dataset_hash': m['dataset_hash'], 'table_hashes': m['table_hashes']}))"
)


def _run(env_update: dict[str, str | None]) -> dict:
    env = dict(os.environ)
    for k, v in env_update.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_same_seed_same_hash_in_process(tiny_ds) -> None:
    again = generate(load_config("tiny"))
    assert again.metadata["dataset_hash"] == tiny_ds.metadata["dataset_hash"]
    assert again.metadata["table_hashes"] == tiny_ds.metadata["table_hashes"]


def test_different_seed_different_data(tiny_ds) -> None:
    other = generate(load_config("tiny", seed=43))
    assert other.metadata["dataset_hash"] != tiny_ds.metadata["dataset_hash"]
    # not just metadata: the event tables themselves differ
    assert other.metadata["table_hashes"]["transactions"] != tiny_ds.metadata["table_hashes"]["transactions"]
    assert other.metadata["table_hashes"]["logins"] != tiny_ds.metadata["table_hashes"]["logins"]


@pytest.mark.slow
def test_pythonhashseed_does_not_change_output(tiny_ds) -> None:
    """Decision 0002: fresh subprocesses with PYTHONHASHSEED=0, =4242 and unset produce identical hashes."""
    runs = {
        "0": _run({"PYTHONHASHSEED": "0"}),
        "4242": _run({"PYTHONHASHSEED": "4242"}),
        "unset": _run({"PYTHONHASHSEED": None}),
    }
    hashes = {k: v["dataset_hash"] for k, v in runs.items()}
    assert len(set(hashes.values())) == 1, hashes
    assert runs["0"]["table_hashes"] == runs["4242"]["table_hashes"] == runs["unset"]["table_hashes"]
    assert hashes["0"] == tiny_ds.metadata["dataset_hash"]
