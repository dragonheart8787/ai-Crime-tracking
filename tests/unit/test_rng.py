"""Stable keys and named streams (decision 0002)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from fcip.common.rng import STREAM_COMPONENTS, keyed_uniform, stable_key, stream
from fcip.common.taxonomy import Archetype, Family

SRC = Path(__file__).resolve().parents[2] / "src" / "fcip"
GUARDED = ("simulation", "common", "temporal")


def test_stable_key_is_blake2b_64_big_endian() -> None:
    import hashlib

    assert stable_key("scenario") == int.from_bytes(
        hashlib.blake2b(b"scenario", digest_size=8).digest(), "big"
    )
    assert 0 <= stable_key("x") < 2**64


def test_no_64bit_collisions_among_stream_names() -> None:
    names = set(STREAM_COMPONENTS) | {a.value for a in Archetype} | {f.value for f in Family}
    keys = [stable_key(n) for n in names]
    assert len(set(keys)) == len(keys)


def test_streams_are_named_not_positional() -> None:
    a = stream(42, "scenario", "fan_in", 3).random(5)
    b = stream(42, "scenario", "fan_in", 3).random(5)
    c = stream(42, "scenario", "fan_in", 4).random(5)
    d = stream(43, "scenario", "fan_in", 3).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)


def test_invalid_stream_components_raise() -> None:
    with pytest.raises(ValueError):
        stream(1, "x", -1)
    with pytest.raises(TypeError):
        stream(1, "x", 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        stream(1, True)


def test_keyed_uniform_in_open_interval_and_deterministic() -> None:
    v = np.arange(10000, dtype=np.int64)
    u = keyed_uniform(stable_key("k"), v)
    assert (u > 0).all() and (u < 1).all()
    assert np.array_equal(u, keyed_uniform(stable_key("k"), v))
    assert abs(float(u.mean()) - 0.5) < 0.02


def test_builtin_hash_is_never_called_in_generator_code() -> None:
    """AST check: no call to the built-in ``hash()`` in simulation, common or temporal code."""
    offenders = []
    for pkg in GUARDED:
        for path in (SRC / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, offenders


def test_ast_guard_detects_a_planted_hash_call(tmp_path: Path) -> None:
    tree = ast.parse("x = hash('a')\n")
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"
        for n in ast.walk(tree)
    )
