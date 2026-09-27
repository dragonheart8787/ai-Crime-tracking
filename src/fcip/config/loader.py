"""Load a generator profile: ``base.yaml`` deep-merged with ``<profile>.yaml`` (decision 0005).

Merge rule: mappings merge recursively; any other value (including lists) in the overlay replaces the
base value. After merging, the optional ``family_defaults`` mapping is deep-merged *under* each family
entry in ``scenarios`` (family-specific values win) and then removed. The resolved mapping is
validated by :class:`GeneratorConfig`; unknown keys raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fcip.common.errors import ConfigError
from fcip.config.generator import GeneratorConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "generator"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def _apply_family_defaults(mapping: dict[str, Any]) -> dict[str, Any]:
    """Expand the optional ``family_defaults`` block into every entry of ``scenarios`` and remove it."""
    defaults = mapping.pop("family_defaults", None)
    if defaults is None:
        return mapping
    scenarios = mapping.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ConfigError("family_defaults given but scenarios is not a mapping")
    mapping["scenarios"] = {fam: deep_merge(defaults, body or {}) for fam, body in scenarios.items()}
    return mapping


def resolve_mapping(
    profile: str, config_dir: Path = CONFIG_DIR, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    merged = deep_merge(_read_yaml(config_dir / "base.yaml"), _read_yaml(config_dir / f"{profile}.yaml"))
    # Top-level keys starting with "_" only hold YAML anchors for reuse inside the file; drop them.
    merged = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    merged["profile"] = profile
    if overrides:
        merged = deep_merge(merged, overrides)
    return _apply_family_defaults(merged)


def load_config(
    profile: str,
    seed: int | None = None,
    config_dir: Path = CONFIG_DIR,
    overrides: dict[str, Any] | None = None,
) -> GeneratorConfig:
    mapping = resolve_mapping(profile, config_dir, overrides)
    if seed is not None:
        mapping = deep_merge(mapping, {"simulation": {"seed": seed}})
    try:
        return GeneratorConfig.model_validate(mapping)
    except Exception as exc:  # re-raised with context; never swallowed
        raise ConfigError(f"invalid generator config for profile {profile!r}: {exc}") from exc
