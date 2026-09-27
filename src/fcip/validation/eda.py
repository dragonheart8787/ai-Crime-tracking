"""Milestone 1 EDA report. Every number in the report is computed here from the dataset on disk; the Markdown is
generated, never edited by hand. Reads oracle tables: validation/EDA code only, never features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fcip.common.frames import to_df
from fcip.common.io import read_table
from fcip.config.generator import GeneratorConfig
from fcip.labels.oracle import OracleLabels
from fcip.temporal.splits import build_example_index
from fcip.temporal.store import TemporalStore
from fcip.validation.sanity import summarize

DAY = 86400


def _md_table(df: pl.DataFrame, floats: int = 3) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:,.{floats}f}")
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compute(data_dir: Path) -> dict[str, Any]:
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = GeneratorConfig.model_validate(meta["config"])
    sm = summarize(data_dir)
    nets = to_df(read_table("ground_truth_networks", data_dir))
    lab = to_df(read_table("labels", data_dir))
    tx = to_df(read_table("transactions", data_dir)).select("ts", "src_account_id", "channel", "txn_type")
    acc = to_df(read_table("accounts", data_dir))
    pt = to_df(read_table("person_truth", data_dir))
    owner = (
        acc.filter(pl.col("account_kind") == "internal")
        .join(pt, left_on="owner_person_id", right_on="person_id")
        .select("account_id", "archetype")
    )
    # timing heterogeneity: hour-of-day and weekend share of outgoing events per archetype
    timing = (
        tx.join(owner, left_on="src_account_id", right_on="account_id")
        .with_columns(
            ((pl.col("ts") % DAY) // 3600).alias("hour"), ((pl.col("ts") // DAY) % 7 >= 5).alias("wkend")
        )
        .group_by("archetype")
        .agg(
            pl.col("hour").mean().alias("mean_hour"),
            pl.col("hour").std().alias("sd_hour"),
            pl.col("wkend").mean().alias("weekend_share"),
        )
        .sort("archetype")
    )
    # instance diversity per family
    div = (
        nets.group_by("family", "is_ood_family")
        .agg(
            pl.len().alias("instances"),
            (
                pl.col("amount_scale_minor").cast(pl.Float64).std()
                / pl.col("amount_scale_minor").cast(pl.Float64).mean()
            ).alias("amount_scale_cv"),
            pl.col("fan_degree").min().alias("fan_min"),
            pl.col("fan_degree").max().alias("fan_max"),
            ((pl.col("end_ts") - pl.col("start_ts")) / 3600).median().alias("median_duration_h"),
            pl.col("n_members").sum().alias("members"),
        )
        .sort("family")
    )
    # label intervals and knowledge
    lab_stats = (
        lab.group_by("phase")
        .agg(
            pl.len().alias("rows"),
            ((pl.col("valid_to") - pl.col("valid_from")) / 3600).median().alias("median_interval_h"),
        )
        .sort("phase")
    )
    anchor = pl.coalesce(pl.col("terminal_ts"), pl.col("end_ts"))
    det = nets.filter(pl.col("detected"))
    latency = (
        ((det["known_at"] - det.select(anchor).to_series()) / DAY).to_numpy() if det.height else np.zeros(0)
    )
    # splits and label knowledge at T_fit
    store = TemporalStore.from_dir(data_dir, meta["sim_end"])
    oracle = OracleLabels.from_dir(data_dir)
    arche = dict(zip(owner["account_id"].to_list(), owner["archetype"].to_list(), strict=True))
    idx = build_example_index(store, oracle, cfg, horizon_s=0, archetype_of=arche)
    tr = idx.train_all
    truth = oracle.targets("active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy())
    known = oracle.training_targets(
        "active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy(), idx.t_fit
    )
    mature = tr["is_mature"].to_numpy()
    splits = {
        "TRAIN points": tr.height,
        "TRAIN ground-truth positives": truth.n_positive,
        "TRAIN positives known at T_fit": int(known.y.sum()),
        "TRAIN latent positives (not yet known at T_fit)": truth.n_positive - int(known.y.sum()),
        "TRAIN mature points (48-day horizon)": int(mature.sum()),
        "VAL points": idx.val.height,
        "TEST points": idx.test.height,
        "OOD pool points": idx.ood.height,
        "OOD-family member accounts": len(idx.e_ood),
        "reference-pool accounts": len(idx.e_ref),
    }
    for name, df in (("VAL", idx.val), ("TEST", idx.test)):
        tg = oracle.targets("active_phase", df["account_id"].to_numpy(), df["t"].to_numpy())
        splits[f"{name} ground-truth positives"] = tg.n_positive
    return {
        "meta": meta,
        "summary": sm,
        "timing": timing,
        "diversity": div,
        "label_phases": lab_stats,
        "latency_days": latency,
        "splits": splits,
    }


def render(r: dict[str, Any]) -> str:
    meta, sm = r["meta"], r["summary"]
    arche = pl.DataFrame(sm["archetypes"]).select(
        "archetype",
        "accounts",
        "out_events_per_acct_per_30d",
        "median_out_amount_minor",
        "p90_out_amount_minor",
        "declined_share_out",
        "digital_share_out",
        "night_share_out",
        "median_distinct_in_cp",
        "max_distinct_in_cp",
    )
    fam = pl.DataFrame(sm["scenario_families"])
    lat = r["latency_days"]
    lat_line = (
        f"median {np.median(lat):.1f}, 5th pct {np.percentile(lat, 5):.1f}, 95th pct {np.percentile(lat, 95):.1f}"
        f" (n = {len(lat)} detected networks)"
        if len(lat)
        else "no detected networks"
    )
    rows = pl.DataFrame({"table": list(meta["row_counts"]), "rows": list(meta["row_counts"].values())})
    status = pl.DataFrame(sm["transactions"]["status"])
    mix = pl.DataFrame(sm["transactions"]["by_type_channel"])
    q = sm["transactions"]["amount_minor_quantiles"]
    sp = pl.DataFrame({"quantity": list(r["splits"]), "value": list(r["splits"].values())})
    roles_line = ", ".join("{} {:,}".format(d["role_label"], d["len"]) for d in sm["label_roles"])
    return f"""# Milestone 1 EDA report

Generated by `python -m fcip.cli eda --data <dir>`; every number below is computed from the dataset on disk.
Oracle tables (archetypes, labels) are used here for description only.

- Profile `{meta["profile"]}`, seed {meta["seed"]}, dataset hash `{meta["dataset_hash"]}`
- Generator commit `{meta["generator"].get("commit")}` ({meta["generator"].get("status")})
- Simulation: {meta["sim_end"] // DAY} days, currency {meta["currency"]["code"]}
  ({meta["currency"]["minor_units_per_major"]} minor units per major)
- Realized prevalence {meta["realized_prevalence"]:.4f} (target {meta["target_prevalence"]}), internal accounts
  {meta["n_internal_accounts"]:,}, recruitment shortfall share {meta["shortfall_share"]:.4f}

## Row counts

{_md_table(rows)}

## Transactions

{_md_table(status)}

Amount quantiles (minor units): 10% {q["0.1"]:,.0f}, 50% {q["0.5"]:,.0f}, 90% {q["0.9"]:,.0f}, 99% {q["0.99"]:,.0f}.

{_md_table(mix)}

## Normal-behavior heterogeneity by archetype (outgoing side)

{_md_table(arche)}

Timing (outgoing events): mean and standard deviation of hour of day, weekend share.

{_md_table(r["timing"])}

Shared IPs: {sm["shared_ip_accounts"]["ips_used_by_2plus_accounts"]:,} IPs are used by two or more accounts; the largest is
shared by {sm["shared_ip_accounts"]["max_accounts_per_ip"]:,} accounts.

## Scenario families

{_md_table(fam)}

Instance diversity (decision 0005):

{_md_table(r["diversity"])}

## Labels

{_md_table(r["label_phases"])}

Roles: {roles_line}.

Detection latency (`known_at` minus terminal event, or network end if none), days: {lat_line}.

## Splits (decision 0009) and label knowledge at T_fit

{_md_table(sp)}
"""


def write_report(data_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(compute(data_dir)), encoding="utf-8")
    return out
