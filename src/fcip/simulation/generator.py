"""Generator orchestration only: population -> recruitment partition -> scenario plans -> normal behavior
-> member suppression -> settlement -> tables and ground truth. No behavior logic lives here.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from fcip import __version__
from fcip.common.errors import ConfigError, InvariantViolation
from fcip.common.hashing import dataset_hash, table_digest
from fcip.common.io import file_sha256, write_table
from fcip.common.rng import keyed_uniform, stable_key
from fcip.common.schemas import TABLE_NAMES, spec
from fcip.common.taxonomy import DeviceKind, Family, RelationType
from fcip.config.generator import GeneratorConfig
from fcip.simulation.behavior import generate_person
from fcip.simulation.events import NO_NETWORK, EventLog
from fcip.simulation.labels import label_rows, member_rows, network_outcome, network_row, realized_prevalence
from fcip.simulation.population import Population, build_population
from fcip.simulation.recruitment import Partition, build_partition
from fcip.simulation.scenarios import InstancePlan, plan_instance
from fcip.simulation.settlement import settle
from fcip.validation.invariants import check_all

EXTERNAL_OPENED_AT = -3650 * 86400


@dataclass
class Dataset:
    tables: dict[str, pa.Table]
    metadata: dict[str, Any]
    plans: list[InstancePlan] = field(default_factory=list)
    partition: Partition | None = None


# ---------------------------------------------------------------- scenario activation
def activate_scenarios(
    cfg: GeneratorConfig, pop: Population, part: Partition
) -> tuple[list[InstancePlan], dict]:
    """Activate instances ``0, 1, ...`` per enabled family until its share of the prevalence quota is met.

    Instance ``i`` is activated if ``i < min_instances`` or if adding it keeps the family closest to its
    quota (``cum + n_members / 2 <= quota``). Reaching ``max_instances`` with the quota unmet raises.
    """
    n_accounts = len(pop.persons)
    enabled = [f for f in Family if cfg.scenarios[f].enabled]
    total_w = sum(cfg.scenarios[f].weight for f in enabled)
    quota_total = cfg.suspicious_prevalence * n_accounts
    plans: list[InstancePlan] = []
    report: dict[str, Any] = {}
    for fam in enabled:
        fcfg = cfg.scenarios[fam]
        quota = quota_total * fcfg.weight / total_w
        cum = 0
        active = 0
        last_size = 0
        for i in range(fcfg.max_instances):
            plan = plan_instance(cfg, pop, part, fam, i)
            last_size = plan.n_members
            if i < fcfg.min_instances or cum + plan.n_members / 2 <= quota:
                plans.append(plan)
                cum += plan.n_members
                active += 1
            else:
                break
        else:
            if quota - cum > max(1, last_size) / 2:
                raise ConfigError(
                    f"{fam.value}: max_instances={fcfg.max_instances} reached with {cum} members, "
                    f"quota {quota:.1f} unmet; raise max_instances"
                )
        report[fam.value] = {"quota_members": round(quota, 2), "members": cum, "instances": active}
    return plans, report


# ---------------------------------------------------------------- member suppression
def apply_suppressions(log: EventLog, plans: list[InstancePlan]) -> tuple[EventLog, int, int]:
    """Thin (or remove) normal events that involve scenario members while they are involved.

    Uses a keyed hash of ``event_id`` per network, never an RNG stream, so the decision for each event is
    independent of every other event and of generation order.
    """
    windows: dict[int, list[tuple[int, int, float, int]]] = {}
    for pl in plans:
        key = stable_key(f"retention/{pl.scenario_id}")
        for sp in pl.suppressions:
            windows.setdefault(sp.account_id, []).append((sp.start, sp.end, sp.retention, key))
    if not windows:
        return log, 0, 0
    accs = np.fromiter(windows.keys(), dtype=np.int64)
    t_src, t_dst = np.asarray(log.t_src, dtype=np.int64), np.asarray(log.t_dst, dtype=np.int64)
    t_ts, t_eid = np.asarray(log.t_ts, dtype=np.int64), np.asarray(log.t_event_id, dtype=np.int64)
    keep_t = np.ones(log.n_txn, dtype=bool)
    for i in np.flatnonzero(np.isin(t_src, accs) | np.isin(t_dst, accs)):
        for acc in (int(t_src[i]), int(t_dst[i])):
            for s, e, ret, key in windows.get(acc, ()):
                if s <= t_ts[i] < e and keyed_uniform(key, t_eid[i : i + 1])[0] >= ret:
                    keep_t[i] = False
    dropped_logins = {log.t_login[i] for i in np.flatnonzero(~keep_t) if log.t_login[i] is not None}
    l_acc, l_ts = np.asarray(log.l_account, dtype=np.int64), np.asarray(log.l_ts, dtype=np.int64)
    l_eid = np.asarray(log.l_event_id, dtype=np.int64)
    linked = {x for x in log.t_login if x is not None}
    keep_l = np.asarray([lid not in dropped_logins for lid in log.l_event_id], dtype=bool)
    for i in np.flatnonzero(np.isin(l_acc, accs)):
        if int(l_eid[i]) in linked:
            continue  # session logins follow their transaction's decision
        for s, e, ret, key in windows.get(int(l_acc[i]), ()):
            if s <= l_ts[i] < e and keyed_uniform(key, l_eid[i : i + 1])[0] >= ret:
                keep_l[i] = False
    return log.filter(keep_t, keep_l), int((~keep_t).sum()), int((~keep_l).sum())


# ---------------------------------------------------------------- table builders
def _entity_tables(cfg: GeneratorConfig, pop: Population, plans: list[InstancePlan]) -> dict[str, pa.Table]:
    sim_end = pop.timebase.sim_end
    created_persons = [p for pl in plans for p in pl.created.persons]
    persons = pop.persons + created_persons
    t: dict[str, pa.Table] = {}
    t["persons"] = pa.table(
        {
            "person_id": [p.pid for p in persons],
            "region": [p.region for p in persons],
            "household_id": [p.household for p in persons],
            "created_at": [p.opened_at for p in persons],
        }
    )
    t["person_truth"] = pa.table(
        {
            "person_id": [p.pid for p in persons],
            "archetype": [p.archetype.value for p in pop.persons]
            + ["scenario_created"] * len(created_persons),
            "origin": ["normal"] * len(pop.persons) + ["scenario"] * len(created_persons),
        }
    )
    acc_rows: list[tuple[Any, ...]] = [
        (
            p.account_id,
            "internal",
            None,
            p.pid,
            p.opened_at,
            None,
            p.overdraft_limit,
            p.initial_balance,
            p.region,
        )
        for p in persons
    ]
    acc_rows += [
        (e.account_id, "external", e.role.value, None, EXTERNAL_OPENED_AT, None, None, 0, None)
        for e in pop.external_accounts
    ]
    cols = [
        "account_id",
        "account_kind",
        "external_role",
        "owner_person_id",
        "opened_at",
        "closed_at",
        "overdraft_limit_minor",
        "initial_balance_minor",
        "region",
    ]
    t["accounts"] = pa.table({c: [r[i] for r in acc_rows] for i, c in enumerate(cols)})
    devices = dict(pop.devices)
    controllers = dict(pop.device_controllers)
    ips = dict(pop.ips)
    for pl in plans:
        devices.update(pl.created.devices)
        controllers.update(pl.created.device_controllers)
        ips.update(pl.created.ips)
    t["devices"] = pa.table(
        {
            "device_id": list(devices),
            "device_kind": [v[0].value for v in devices.values()],
            "first_seen_at": [v[1] for v in devices.values()],
        }
    )
    t["ips"] = pa.table(
        {
            "ip_id": list(ips),
            "ip_context": [v[0].value for v in ips.values()],
            "region": [v[1] for v in ips.values()],
        }
    )
    t["atms"] = pa.table({"atm_id": list(pop.atms), "region": list(pop.atms.values())})
    t["merchants"] = pa.table(
        {
            "merchant_id": list(pop.merchants),
            "merchant_category": [v[0] for v in pop.merchants.values()],
            "region": [v[1] for v in pop.merchants.values()],
        }
    )
    rel: list[tuple[str, int, int, int, int]] = []
    for p in persons:
        rel.append((RelationType.OWNS_ACCOUNT.value, p.pid, p.account_id, p.opened_at, sim_end))
    for dev, owners in controllers.items():
        for o in owners:
            rel.append((RelationType.CONTROLS.value, o, dev, devices[dev][1], sim_end))
    uses: set[tuple[int, int]] = set()
    for p in pop.persons:
        home = pop.households[p.household]["ip"]
        for dev, kind in p.devices:
            uses.add((dev, home))
            if p.workplace_ip is not None and kind != DeviceKind.TABLET:
                uses.add((dev, p.workplace_ip))
    for pl in plans:
        uses.update(pl.created.uses_ip)
    for dev, ip in sorted(uses):
        rel.append((RelationType.USES_IP.value, dev, ip, devices[dev][1], sim_end))
    t["relations"] = pa.table(
        {
            c: [r[i] for r in rel]
            for i, c in enumerate(["relation_type", "src_id", "dst_id", "valid_from", "valid_to"])
        }
    )
    return t


def _git_commit() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "status": f"unavailable: {type(exc).__name__}"}
    return {"commit": head.stdout.strip(), "status": "dirty" if dirty.stdout.strip() else "clean"}


def _package_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "fcip": __version__}
    for pkg in ("numpy", "pyarrow", "polars", "pydantic", "pyyaml"):
        out[pkg] = importlib.metadata.version(pkg)
    return out


# ---------------------------------------------------------------- main entry
def generate(cfg: GeneratorConfig) -> Dataset:
    pop = build_population(cfg)
    sim_end = pop.timebase.sim_end
    part = build_partition(cfg, [p.account_id for p in pop.persons])
    plans, activation = activate_scenarios(cfg, pop, part)

    normal = EventLog()
    for p in pop.persons:
        normal.extend(generate_person(pop, cfg, p))
    normal, n_sup_txn, n_sup_login = apply_suppressions(normal, plans)

    full = EventLog()
    full.extend(normal)
    for pl in plans:
        full.extend(pl.log)

    # ledger setup
    initial = {p.account_id: p.initial_balance for p in pop.persons}
    limits: dict[int, int | None] = {p.account_id: p.overdraft_limit for p in pop.persons}
    for pl in plans:
        for cp in pl.created.persons:
            initial[cp.account_id] = 0
            limits[cp.account_id] = 0
    for ext in pop.external_accounts:
        initial[ext.account_id] = 0
        limits[ext.account_id] = None
    st = settle(
        full.t_event_id, full.t_ts, full.t_src, full.t_dst, full.t_amount, full.t_depends_on, initial, limits
    )

    order = st.order
    tx = {
        "event_id": [full.t_event_id[i] for i in order],
        "ts": [full.t_ts[i] for i in order],
        "txn_type": [full.t_type[i] for i in order],
        "channel": [full.t_channel[i] for i in order],
        "src_account_id": [full.t_src[i] for i in order],
        "dst_account_id": [full.t_dst[i] for i in order],
        "merchant_id": [full.t_merchant[i] for i in order],
        "atm_id": [full.t_atm[i] for i in order],
        "amount_minor": [full.t_amount[i] for i in order],
        "region": [full.t_region[i] for i in order],
        "device_id": [full.t_device[i] for i in order],
        "ip_id": [full.t_ip[i] for i in order],
        "login_event_id": [full.t_login[i] for i in order],
        "status": st.status[order].tolist(),
        "src_balance_before_minor": st.src_before[order],
        "src_balance_after_minor": st.src_after[order],
        "dst_balance_before_minor": st.dst_before[order],
        "dst_balance_after_minor": st.dst_after[order],
    }
    lorder = np.lexsort((np.asarray(full.l_event_id, dtype=np.int64), np.asarray(full.l_ts, dtype=np.int64)))
    lg = {
        "event_id": [full.l_event_id[i] for i in lorder],
        "ts": [full.l_ts[i] for i in lorder],
        "account_id": [full.l_account[i] for i in lorder],
        "device_id": [full.l_device[i] for i in lorder],
        "ip_id": [full.l_ip[i] for i in lorder],
        "channel": [full.l_channel[i] for i in lorder],
        "outcome": [full.l_outcome[i] for i in lorder],
    }
    tables = _entity_tables(cfg, pop, plans)
    tables["transactions"] = pa.table(tx)
    tables["logins"] = pa.table(lg)

    # ground truth
    net_txn: dict[int, list[tuple[int, int, str | None, str]]] = {}
    for i in range(full.n_txn):
        if full.t_network[i] != NO_NETWORK:
            net_txn.setdefault(full.t_network[i], []).append(
                (full.t_event_id[i], full.t_ts[i], full.t_phase[i], st.status[i])
            )
    lab, ev, nets, mem = [], [], [], []
    high_risk = {ph.value for ph in cfg.high_risk_phases}
    for pl in plans:
        out = network_outcome(cfg, pl, net_txn.get(pl.network_id, []), sim_end)
        lab += label_rows(pl, out, sim_end)
        nets.append(network_row(pl, out))
        mem += member_rows(pl)
        for i in range(pl.log.n_txn):
            ev_id = pl.log.t_event_id[i]
            ev.append(
                (
                    "transactions",
                    ev_id,
                    pl.scenario_id,
                    pl.network_id,
                    pl.family.value,
                    pl.log.t_phase[i],
                    pl.log.t_event_type[i],
                    pl.log.t_phase[i] in high_risk,
                    ev_id == out.terminal_event_id,
                    pl.is_ood_family,
                    out.known_at,
                )
            )
        for i in range(pl.log.n_login):
            ev.append(
                (
                    "logins",
                    pl.log.l_event_id[i],
                    pl.scenario_id,
                    pl.network_id,
                    pl.family.value,
                    pl.log.l_phase[i],
                    pl.log.l_event_type[i],
                    False,
                    False,
                    pl.is_ood_family,
                    out.known_at,
                )
            )
    lab_cols = list(spec("labels").schema.names)
    tables["labels"] = pa.table({c: [r[c] for r in lab] for c in lab_cols}, schema=spec("labels").schema)
    ev_cols = list(spec("event_labels").schema.names)
    tables["event_labels"] = pa.table(
        {c: [r[i] for r in ev] for i, c in enumerate(ev_cols)}, schema=spec("event_labels").schema
    )
    net_cols = list(spec("ground_truth_networks").schema.names)
    tables["ground_truth_networks"] = pa.table(
        {c: [r[c] for r in nets] for c in net_cols}, schema=spec("ground_truth_networks").schema
    )
    mem_cols = list(spec("network_members").schema.names)
    tables["network_members"] = pa.table(
        {c: [r[c] for r in mem] for c in mem_cols}, schema=spec("network_members").schema
    )

    missing = set(TABLE_NAMES) - set(tables)
    if missing:
        raise InvariantViolation(f"generator did not produce tables {sorted(missing)}")
    checks_passed = check_all(
        tables,
        sim_end=sim_end,
        session_window=cfg.infrastructure.session_window_seconds,
        high_risk_phases=[ph.value for ph in cfg.high_risk_phases],
    )
    digests = {name: table_digest(name, tables[name]) for name in TABLE_NAMES}
    n_internal = len(pop.persons) + sum(pl.n_created for pl in plans)
    metadata: dict[str, Any] = {
        "format": "fcip-dataset-v1",
        "profile": cfg.profile,
        "seed": cfg.simulation.seed,
        "config": json.loads(cfg.canonical_json()),
        "config_hash": cfg.config_hash(),
        "currency": cfg.currency.model_dump(),
        "sim_end": sim_end,
        "generator": {"version": __version__, **_git_commit()},
        "package_versions": _package_versions(),
        "table_hashes": {k: v.hex() for k, v in digests.items()},
        "dataset_hash": dataset_hash(digests),
        "hash_algorithm": "fcip-content-hash-v1",
        "oracle_tables": sorted(n for n in TABLE_NAMES if spec(n).oracle),
        "row_counts": {k: tables[k].num_rows for k in TABLE_NAMES},
        "activation": activation,
        "realized_prevalence": realized_prevalence(plans, n_internal),
        "target_prevalence": cfg.suspicious_prevalence,
        "n_internal_accounts": n_internal,
        "recruitment_shortfall": {pl.scenario_id: pl.n_created for pl in plans if pl.n_created},
        "shortfall_share": (sum(pl.n_created for pl in plans) / max(1, sum(pl.n_members for pl in plans))),
        "suppressed_normal_events": {"transactions": n_sup_txn, "logins": n_sup_login},
        "invariant_checks_passed": checks_passed,
    }
    if metadata["shortfall_share"] > cfg.recruitment.max_shortfall_share:
        raise InvariantViolation(
            f"recruitment shortfall share {metadata['shortfall_share']:.3f} exceeds "
            f"max_shortfall_share={cfg.recruitment.max_shortfall_share}"
        )
    return Dataset(tables=tables, metadata=metadata, plans=plans, partition=part)


def write_dataset(ds: Dataset, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in TABLE_NAMES:
        path = write_table(name, ds.tables[name], out_dir)
        files[name] = file_sha256(path)
    meta = dict(ds.metadata)
    meta["file_sha256"] = files
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta
