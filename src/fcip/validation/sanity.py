"""Short data-shape sanity summary (Milestone 1 checkpoint 1): row counts, per-archetype behavior,
scenario family counts. Every number is computed from the dataset on disk; nothing is hard-coded.

This reads oracle tables (``person_truth``, label tables) and is for validation/EDA only, never features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from fcip.common.io import read_table
from fcip.common.schemas import TABLE_NAMES


def _load(data_dir: Path) -> dict[str, pl.DataFrame]:
    return {n: pl.from_arrow(read_table(n, data_dir)) for n in TABLE_NAMES}  # type: ignore[misc]


def summarize(data_dir: Path) -> dict[str, Any]:
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    d = _load(data_dir)
    t, acc, pt = d["transactions"], d["accounts"], d["person_truth"]
    sim_days = meta["sim_end"] / 86400
    owner = (
        acc.filter(pl.col("account_kind") == "internal")
        .join(pt, left_on="owner_person_id", right_on="person_id")
        .select("account_id", "archetype")
    )
    out_side = t.join(owner, left_on="src_account_id", right_on="account_id").select(
        "archetype", "src_account_id", "amount_minor", "status", "channel", "txn_type", "ts"
    )
    in_side = t.join(owner, left_on="dst_account_id", right_on="account_id").select(
        "archetype", "dst_account_id", "amount_minor", "status"
    )
    per_acc_out = out_side.group_by("archetype", "src_account_id").agg(pl.len().alias("n_out"))
    per_acc_in = in_side.group_by("archetype", "dst_account_id").agg(pl.len().alias("n_in"))
    arche = (
        owner.group_by("archetype")
        .len()
        .rename({"len": "accounts"})
        .join(
            per_acc_out.group_by("archetype").agg((pl.col("n_out").sum() / sim_days * 30).alias("_tot")),
            on="archetype",
            how="left",
        )
        .join(
            out_side.group_by("archetype").agg(
                pl.col("amount_minor").median().alias("median_out_amount_minor"),
                pl.col("amount_minor").quantile(0.9).alias("p90_out_amount_minor"),
                (pl.col("status") == "DECLINED").mean().alias("declined_share_out"),
                pl.col("channel").is_in(["APP", "WEB"]).mean().alias("digital_share_out"),
                ((pl.col("ts") % 86400) < 5 * 3600).mean().alias("night_share_out"),
            ),
            on="archetype",
            how="left",
        )
        .join(
            per_acc_in.group_by("archetype").agg(pl.col("n_in").median().alias("median_in_events_per_acct")),
            on="archetype",
            how="left",
        )
        .with_columns((pl.col("_tot") / pl.col("accounts")).alias("out_events_per_acct_per_30d"))
        .drop("_tot")
        .sort("archetype")
    )
    cp_out = (
        t.join(owner, left_on="src_account_id", right_on="account_id")
        .group_by("archetype", "src_account_id")
        .agg(pl.col("dst_account_id").n_unique().alias("distinct_out_cp"))
    )
    cp_in = (
        t.join(owner, left_on="dst_account_id", right_on="account_id")
        .group_by("archetype", "dst_account_id")
        .agg(pl.col("src_account_id").n_unique().alias("distinct_in_cp"))
    )
    arche = arche.join(
        cp_out.group_by("archetype").agg(pl.col("distinct_out_cp").median().alias("median_distinct_out_cp")),
        on="archetype",
        how="left",
    ).join(
        cp_in.group_by("archetype").agg(
            pl.col("distinct_in_cp").median().alias("median_distinct_in_cp"),
            pl.col("distinct_in_cp").max().alias("max_distinct_in_cp"),
        ),
        on="archetype",
        how="left",
    )
    nets, mem, lab = d["ground_truth_networks"], d["network_members"], d["labels"]
    fam = (
        nets.group_by("family", "is_ood_family")
        .agg(
            pl.len().alias("instances"),
            pl.col("n_members").sum().alias("members"),
            pl.col("n_created_accounts").sum().alias("created_accounts"),
            pl.col("terminal_ts").is_not_null().sum().alias("terminated"),
            pl.col("censored").sum().alias("censored"),
            pl.col("detected").sum().alias("detected"),
            pl.col("amount_scale_minor").median().alias("median_amount_scale_minor"),
        )
        .sort("family")
    )
    roles = lab.group_by("role_label").len().sort("role_label")
    scen_evt = d["event_labels"].group_by("event_table").len()
    return {
        "dataset_hash": meta["dataset_hash"],
        "profile": meta["profile"],
        "seed": meta["seed"],
        "row_counts": meta["row_counts"],
        "transactions": {
            "status": t.group_by("status").len().sort("status").to_dicts(),
            "by_type_channel": t.group_by("txn_type", "channel")
            .len()
            .sort("len", descending=True)
            .to_dicts(),
            "amount_minor_quantiles": {
                q: float(t["amount_minor"].quantile(q) or 0.0) for q in (0.1, 0.5, 0.9, 0.99)
            },
        },
        "logins": d["logins"].group_by("outcome").len().sort("outcome").to_dicts(),
        "archetypes": arche.to_dicts(),
        "scenario_families": fam.to_dicts(),
        "label_roles": roles.to_dicts(),
        "scenario_event_rows": scen_evt.sort("event_table").to_dicts(),
        "prevalence": {
            "target": meta["target_prevalence"],
            "realized": meta["realized_prevalence"],
            "n_internal_accounts": meta["n_internal_accounts"],
            "member_accounts": mem["entity_id"].n_unique(),
        },
        "shortfall_share": meta["shortfall_share"],
        "suppressed_normal_events": meta["suppressed_normal_events"],
        "ip_contexts": d["ips"].group_by("ip_context").len().sort("ip_context").to_dicts(),
        "shared_ip_accounts": (
            d["logins"]
            .group_by("ip_id")
            .agg(pl.col("account_id").n_unique().alias("n"))
            .select(
                (pl.col("n") > 1).sum().alias("ips_used_by_2plus_accounts"),
                pl.col("n").max().alias("max_accounts_per_ip"),
            )
        ).to_dicts()[0],
    }
