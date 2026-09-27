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
    """``txn_rows``: (event_id, ts, phase, status) of this network's transactions.

    Terminal event: the last settled EXIT event. Detection anchor (decision 0004, revised at Milestone 1
    checkpoint 2): the latest of the terminal event, the network's last scenario event of any status (declined
    attempts and logins are observable activity), and the start of its latest label interval. Detection
    therefore never precedes any observed activity or labelled state of the network.
    """
    exits = [
        (ts, eid) for eid, ts, ph, st in txn_rows if ph == Phase.EXIT.value and st == Status.SETTLED.value
    ]
    t_ts, t_id = max(exits) if exits else (None, None)
    last_event = max(list(plan.log.t_ts) + list(plan.log.l_ts), default=plan.start_ts)
    starts = [s for rows in member_intervals(plan, sim_end).values() for _k, _p, _r, s, _e in rows]
    anchor = max([last_event, max(starts, default=plan.start_ts)] + ([t_ts] if t_ts is not None else []))
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


def first_own_event_ts(plan: InstancePlan) -> dict[int, int]:
    """Earliest scenario event of this network that each account appears in: as source or destination of a
    transaction (settled or declined; an attempt is observable) or as the account of a login."""
    first: dict[int, int] = {}
    log = plan.log
    for ts, src, dst in zip(log.t_ts, log.t_src, log.t_dst, strict=True):
        for acc in (src, dst):
            if ts < first.get(acc, ts + 1):
                first[acc] = ts
    for ts, acc in zip(log.l_ts, log.l_account, strict=True):
        if ts < first.get(acc, ts + 1):
            first[acc] = ts
    return first


def member_intervals(plan: InstancePlan, sim_end: int) -> dict[int, list[tuple[int, Phase, Role, int, int]]]:
    """Label intervals per member (A1, Milestone 1 checkpoint 2).

    A member's labels start at its own first observable event in the network, never at a network phase
    boundary before that: phase intervals that end at or before the first own event are dropped, the one
    containing it starts at it, later ones keep their phase start (the account's own history already shows
    participation). A member without any own event gets no label rows.
    """
    first = first_own_event_ts(plan)
    out: dict[int, list[tuple[int, Phase, Role, int, int]]] = {}
    for m in plan.members:
        f = first.get(m.account_id)
        rows: list[tuple[int, Phase, Role, int, int]] = []
        if f is not None:
            for k, role in sorted(m.roles.items(), key=lambda kv: plan.intervals[kv[0]][1]):
                phase, s, e = plan.intervals[k]
                e = min(e, sim_end)
                if e <= f or s >= sim_end:
                    continue
                s = max(s, f)
                if e > s:
                    rows.append((k, phase, role, s, e))
        out[m.account_id] = rows
    return out


def label_rows(plan: InstancePlan, outcome: NetworkOutcome, sim_end: int) -> list[dict]:
    rows = []
    for acc, intervals in member_intervals(plan, sim_end).items():
        for _k, phase, role, s, e in intervals:
            rows.append(
                {
                    "entity_type": "account",
                    "entity_id": acc,
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
    """``joined_ts`` is the member's first own event in the network (A1); ``first_role`` is the role of its
    first labelled interval. Members without an own event keep the start of their first role interval."""
    first = first_own_event_ts(plan)
    rows = []
    for m in plan.members:
        f = first.get(m.account_id)
        ordered = sorted(m.roles.items(), key=lambda kv: plan.intervals[kv[0]][1])
        if f is not None:
            live = [kv for kv in ordered if plan.intervals[kv[0]][2] > f]
            role = (live or ordered)[0][1] if ordered else Role.SUSPICIOUS_UNKNOWN
            joined = f
        else:
            role = ordered[0][1] if ordered else Role.SUSPICIOUS_UNKNOWN
            joined = m.joined_ts if m.joined_ts is not None else plan.start_ts
        rows.append(
            {
                "network_id": plan.network_id,
                "entity_type": "account",
                "entity_id": m.account_id,
                "first_role": role.value,
                "joined_ts": joined,
                "is_created": m.is_created,
            }
        )
    return rows


def realized_prevalence(plans: list[InstancePlan], n_internal_accounts: int) -> float:
    members = {m.account_id for pl in plans for m in pl.members}
    return len(members) / n_internal_accounts if n_internal_accounts else float("nan")
