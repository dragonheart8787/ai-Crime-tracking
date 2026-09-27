"""Named, hierarchical RNG streams (decision 0002).

A stream is identified by a *name path*, never by position, so adding or removing a component does not
shift any other component's randomness. Python's built-in ``hash()`` must never be used for anything
that influences generated data (it is salted per process); ``stable_key`` is the only allowed hash.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Registry of every string component used in a stream path. A unit test checks it for 64-bit
# collisions. Adding a new component name means adding it here.
STREAM_COMPONENTS: frozenset[str] = frozenset(
    {
        "population",
        "persons",
        "households",
        "accounts",
        "devices",
        "ips",
        "atms",
        "merchants",
        "external",
        "contacts",
        "employment",
        "landlords",
        "normal",
        "scenario",
        "label_latency",
        "recruit",
        "retention",
        "logins",
    }
)


def stable_key(name: str) -> int:
    """64-bit key from blake2b over the UTF-8 name, big-endian. Identical on every process and machine."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")


def stream(seed: int, *path: str | int) -> np.random.Generator:
    """Return an independent generator for ``seed`` and a name path of strings and non-negative ints."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    key: list[int] = []
    for part in path:
        if isinstance(part, bool):
            raise TypeError("bool is not a valid stream path component")
        if isinstance(part, str):
            key.append(stable_key(part))
        elif isinstance(part, int | np.integer):
            if part < 0:
                raise ValueError(f"integer stream path components must be >= 0, got {part}")
            key.append(int(part))
        else:
            raise TypeError(f"invalid stream path component {part!r}")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(key)))


_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    z = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def unit_uniform(x: np.ndarray) -> np.ndarray:
    """Map uint64 hash values to floats strictly inside (0, 1)."""
    top53 = (np.asarray(x, dtype=np.uint64) >> np.uint64(11)).astype(np.float64)
    return (top53 + 0.5) / float(2**53)


def keyed_uniform(key: int, values: np.ndarray) -> np.ndarray:
    """Deterministic uniforms in (0, 1) for integer ``values`` under a 64-bit ``key`` (no RNG state)."""
    v = np.asarray(values, dtype=np.int64).astype(np.uint64)
    return unit_uniform(splitmix64(splitmix64(v ^ np.uint64(key & 0xFFFFFFFFFFFFFFFF))))
