"""Import and access boundaries (decision 0003).

* Feature code (``fcip/features``) and the temporal core (``fcip/temporal``, except the split protocol module)
  must not import the oracle, the generator, or validation code, and must not name oracle tables.
* Outside ``fcip/temporal``, no code may reach into private attributes of other objects (``x._name`` where
  ``x`` is not ``self``/``cls``): the store's internals are only reachable through its public API.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "fcip"
FORBIDDEN_IMPORTS = ("fcip.labels.oracle", "fcip.simulation", "fcip.validation")
ORACLE_TABLES = ("person_truth", "event_labels", "ground_truth_networks", "network_members")
PROTOCOL_MODULES = {"temporal/splits.py"}  # may read the oracle for protocol (which entity goes where)


def _modules(sub: str) -> list[Path]:
    return sorted((SRC / sub).rglob("*.py"))


def _imports(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_feature_and_temporal_code_does_not_import_ground_truth() -> None:
    offenders = []
    for sub in ("features", "temporal"):
        for path in _modules(sub):
            rel = str(path.relative_to(SRC))
            if rel in PROTOCOL_MODULES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for mod in _imports(tree):
                if mod.startswith(FORBIDDEN_IMPORTS):
                    offenders.append(f"{rel} imports {mod}")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in ORACLE_TABLES
                ):
                    offenders.append(f"{rel} names oracle table {node.value!r}")
    assert not offenders, offenders


def test_no_private_access_to_other_objects_outside_temporal() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        if rel.startswith("temporal/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"))
            ):
                offenders.append(f"{rel}:{node.lineno} .{node.attr}")
    assert not offenders, offenders


def test_boundary_checks_detect_violations() -> None:
    bad = ast.parse("from fcip.labels.oracle import OracleLabels\nx = store._s\n")
    assert any(m.startswith(FORBIDDEN_IMPORTS) for m in _imports(bad))
    assert any(isinstance(n, ast.Attribute) and n.attr == "_s" for n in ast.walk(bad))
