"""Command-line entry point: ``python -m fcip.cli <command>``.

Commands: ``simulate``, ``validate``, ``sanity``, ``eda``, ``gate``, ``export-csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fcip.common.io import export_csv, read_table
from fcip.common.schemas import TABLE_NAMES
from fcip.config.loader import load_config


def _simulate(args: argparse.Namespace) -> int:
    from fcip.simulation.generator import generate, write_dataset

    cfg = load_config(args.profile, seed=args.seed)
    out = Path(args.out) if args.out else Path("data/processed") / f"{args.profile}_seed{cfg.simulation.seed}"
    t0 = time.perf_counter()
    ds = generate(cfg)
    meta = write_dataset(ds, out)
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "dataset_hash": meta["dataset_hash"],
                "row_counts": meta["row_counts"],
                "realized_prevalence": meta["realized_prevalence"],
                "invariant_checks_passed": meta["invariant_checks_passed"],
                "seconds": round(time.perf_counter() - t0, 1),
            },
            indent=2,
        )
    )
    if args.csv:
        for name in TABLE_NAMES:
            export_csv(name, out, out / "csv")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from fcip.common.hashing import dataset_hash, table_digest
    from fcip.validation.invariants import check_all

    data = Path(args.data)
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    tables = {n: read_table(n, data) for n in TABLE_NAMES}
    digests = {n: table_digest(n, tables[n]) for n in TABLE_NAMES}
    recomputed = dataset_hash(digests)
    if recomputed != meta["dataset_hash"]:
        print(
            f"dataset hash mismatch: metadata {meta['dataset_hash']} recomputed {recomputed}", file=sys.stderr
        )
        return 1
    from fcip.config.generator import GeneratorConfig

    cfg = GeneratorConfig.model_validate(meta["config"])
    ran = check_all(
        tables,
        sim_end=meta["sim_end"],
        session_window=cfg.infrastructure.session_window_seconds,
        high_risk_phases=[p.value for p in cfg.high_risk_phases],
        min_cv=cfg.validation.min_instance_amount_cv,
        min_instances=cfg.validation.min_instances_for_cv,
    )
    print(
        json.dumps(
            {"dataset_hash": recomputed, "hash_matches_metadata": True, "checks_passed": ran}, indent=2
        )
    )
    return 0


def _sanity(args: argparse.Namespace) -> int:
    from fcip.validation.sanity import summarize

    print(json.dumps(summarize(Path(args.data)), indent=2, default=str))
    return 0


def _eda(args: argparse.Namespace) -> int:
    from fcip.validation.eda import write_report

    print(write_report(Path(args.data), Path(args.out)))
    return 0


def _gate(args: argparse.Namespace) -> int:
    from fcip.validation.gate import evaluate, run_seed

    results = []
    for d in args.data:
        t0 = time.perf_counter()
        r = run_seed(Path(d), ref_pool_method=args.ref_pool_method)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(
            json.dumps({"seed": r["seed"], "G1": r["G1"]["best_ap"], "G2": r["G2"], "pi": r["pi"]}),
            flush=True,
        )
    out = {"experiment": "EXP-M1-G", "per_seed": results, "criteria": evaluate(results)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out["criteria"], indent=2))
    return 0


def _export(args: argparse.Namespace) -> int:
    data = Path(args.data)
    for name in TABLE_NAMES:
        print(export_csv(name, data, data / "csv"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fcip")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("simulate", help="generate a synthetic dataset")
    p.add_argument("--profile", required=True, help="config profile: tiny | dev | research")
    p.add_argument("--seed", type=int, default=None, help="override simulation.seed")
    p.add_argument(
        "--out", default=None, help="output directory (default data/processed/<profile>_seed<seed>)"
    )
    p.add_argument("--csv", action="store_true", help="also export every table as CSV")
    p.set_defaults(fn=_simulate)
    p = sub.add_parser("validate", help="recompute the dataset hash and run all invariant checks")
    p.add_argument("--data", required=True)
    p.set_defaults(fn=_validate)
    p = sub.add_parser("sanity", help="print the data-shape sanity summary")
    p.add_argument("--data", required=True)
    p.set_defaults(fn=_sanity)
    p = sub.add_parser("eda", help="write the Milestone 1 EDA report for a dataset")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="reports/eda_milestone1.md")
    p.set_defaults(fn=_eda)
    p = sub.add_parser("gate", help="run the pre-registered EXP-M1-G gates on calibration-seed datasets")
    p.add_argument("--data", required=True, nargs="+", help="one dataset directory per calibration seed")
    p.add_argument("--out", default="reports/gate_exp_m1_g.json")
    p.add_argument("--ref-pool-method", default=None, choices=["none_cell_hash", "stratified_archetype"],
                   help="override the reference-pool method (A2 decision rule, EXPERIMENTS.md freeze item 6)")
    p.set_defaults(fn=_gate)
    p = sub.add_parser("export-csv", help="export every table of a dataset as CSV")
    p.add_argument("--data", required=True)
    p.set_defaults(fn=_export)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
