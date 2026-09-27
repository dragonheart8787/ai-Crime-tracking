"""Ground-truth label tables (decision 0004): interval ``labels``, ``event_labels``, networks, members.

Roles are phase-dependent: a member's role in each phase interval comes from the scenario phase at that
time, never from its final role. ``known_at`` follows the network-level retrospective detection model
(simplification S4): one draw per network, identical for every row of that network.
"""

from __future__ import annotations

from dataclasses import dataclass

from fcip.common.rng import stream
from fcip.common.taxonomy import Phase, Role, Status
from fcip.common.timebase import SECONDS_PER_DAY
from fcip.config.generator import GeneratorConfig
from fcip.simulation.scenarios import InstancePlan


@dataclass
class NetworkOutcome:
    terminal_event_id: int | None
    terminal_ts: int | None
    anchor_ts: int
    known_at: int | None
    detected: bool


def network_outcome(
    cfg: GeneratorConfig, plan: InstancePlan, txn_rows: list[tuple[int, int, str | None, str]], sim_end: int
) -> NetworkOutcome:
    """``txn_rows``: (event_id, ts, phase, status) of this network's transactions."""
    exits = [
        (ts, eid) for eid, ts, ph, st in txn_rows if ph == Phase.EXIT.value and st == Status.SETTLED.value
    ]
    settled = [(ts, eid) for eid, ts, ph, st in txn_rows if st == Status.SETTLED.value]
    if exits:
        t_ts, t_id = max(exits)
        anchor = t_ts
    else:
        t_ts, t_id = None, None
        anchor = max(settled)[0] if settled else plan.start_ts
    rng = stream(cfg.simulation.seed, "label_latency", plan.family.value, plan.instance_idx)
    u_never = float(rng.random())
    latency_days = float(cfg.label_knowledge.latency_days.sample(rng))
    lk = cfg.label_knowledge
    if lk.regime == "oracle":
        known_at: int | None = anchor
    elif u_never < lk.p_never_known:
        known_at = None
    else:
        known_at = anchor + int(round(latency_days * SECONDS_PER_DAY))
        if known_at >= sim_end:
            known_at = None
    return NetworkOutcome(t_id, t_ts, anchor, known_at, known_at is not None)


def label_rows(plan: InstancePlan, outcome: NetworkOutcome, sim_end: int) -> list[dict]:
    rows = []
    for m in plan.members:
        for k, role in sorted(m.roles.items()):
            phase, s, e = plan.intervals[k]
            if s >= sim_end or e <= s:
                continue
            rows.append(
                {
                    "entity_type": "account",
                    "entity_id": m.account_id,
                    "valid_from": s,
                    "valid_to": e,
                    "risk_label": 0 if role == Role.VICTIM_LIKE else 1,
                    "role_label": role.value,
                    "scenario_id": plan.scenario_id,
                    "network_id": plan.network_id,
                    "family": plan.family.value,
                    "phase": phase.value,
                    "is_ood_family": plan.is_ood_family,
                    "known_at": outcome.known_at,
                }
            )
    return rows


def network_row(plan: InstancePlan, outcome: NetworkOutcome) -> dict:
    p = plan.params
    return {
        "network_id": plan.network_id,
        "scenario_id": plan.scenario_id,
        "family": plan.family.value,
        "instance_idx": plan.instance_idx,
        "is_ood_family": plan.is_ood_family,
        "start_ts": plan.start_ts,
        "end_ts": plan.end_ts,
        "terminal_ts": outcome.terminal_ts,
        "censored": plan.censored,
        "n_members": plan.n_members,
        "n_created_accounts": plan.n_created,
        "n_rounds": plan.n_rounds,
        "detected": outcome.detected,
        "known_at": outcome.known_at,
        "amount_scale_minor": int(round(p["amount_scale"])),
        "fan_degree": int(p["fan_degree"]),
        "hops": int(p["hops"]),
        "holding_seconds": int(round(p["holding_hours"] * 3600)),
        "retention_permille": int(round(p["retention"] * 1000)),
        "forward_permille": int(round(p["forward_fraction"] * 1000)),
    }


def member_rows(plan: InstancePlan) -> list[dict]:
    rows = []
    for m in plan.members:
        first = min(m.roles.items(), key=lambda kv: plan.intervals[kv[0]][1]) if m.roles else None
        rows.append(
            {
                "network_id": plan.network_id,
                "entity_type": "account",
                "entity_id": m.account_id,
                "first_role": first[1].value if first else Role.SUSPICIOUS_UNKNOWN.value,
                "joined_ts": m.joined_ts if m.joined_ts is not None else plan.start_ts,
                "is_created": m.is_created,
            }
        )
    return rows


def realized_prevalence(plans: list[InstancePlan], n_internal_accounts: int) -> float:
    members = {m.account_id for pl in plans for m in pl.members}
    return len(members) / n_internal_accounts if n_internal_accounts else float("nan")
