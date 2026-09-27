"""Suspicious scenario families: abstract detection-research patterns (CLAUDE.md 2.1, decision 0005).

Parameters describe pattern *shape* only (degrees, hop counts, durations, amount-distribution shape,
timing). Nothing here reads a detector, a score or a threshold, and nothing searches parameters against
detection. Each instance ``(family, i)`` uses only its own stream ``("scenario", family, i)``, recruits
only from its own recruitment cell (decision 0002), and allocates IDs only in its own namespace.

Every instance follows the same stochastic phase grammar (decision 0004):
``SETUP -> (INFLOW -> HOLD) x rounds -> MOVEMENT -> EXIT | INACTIVE``, where the number of rounds and
whether the network exits are random; families differ in topology and in which phases carry events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from fcip.common import ids
from fcip.common.rng import stream
from fcip.common.taxonomy import (
    FAMILY_CODE,
    Channel,
    DeviceKind,
    EventType,
    ExternalRole,
    Family,
    IpContext,
    LoginOutcome,
    Phase,
    Role,
    TxnType,
)
from fcip.common.timebase import SECONDS_PER_DAY, SECONDS_PER_HOUR
from fcip.config.distributions import sample_int
from fcip.config.generator import FamilyCfg, GeneratorConfig
from fcip.simulation.behavior import START_MARGIN_S
from fcip.simulation.events import EventLog
from fcip.simulation.logins import emit_session_login
from fcip.simulation.population import Person, Population
from fcip.simulation.recruitment import Partition


@dataclass
class Member:
    account_id: int
    person: Person
    is_created: bool
    slot: str
    roles: dict[int, Role] = field(default_factory=dict)  # phase-interval index -> role
    joined_ts: int | None = None


@dataclass
class CreatedEntities:
    persons: list[Person] = field(default_factory=list)
    devices: dict[int, tuple[DeviceKind, int]] = field(default_factory=dict)
    device_controllers: dict[int, list[int]] = field(default_factory=dict)
    ips: dict[int, tuple[IpContext, int]] = field(default_factory=dict)
    uses_ip: list[tuple[int, int]] = field(default_factory=list)  # (device_id, ip_id) assignments


@dataclass
class Suppression:
    """Window in which a member's normal events are thinned (retention) or removed (retention 0)."""

    account_id: int
    start: int
    end: int
    retention: float


@dataclass
class InstancePlan:
    family: Family
    instance_idx: int
    network_id: int
    scenario_id: str
    is_ood_family: bool
    params: dict[str, float]
    members: list[Member]
    intervals: list[tuple[Phase, int, int]]  # clipped to sim_end; only phases that occur
    log: EventLog
    created: CreatedEntities
    suppressions: list[Suppression]
    start_ts: int
    end_ts: int  # planned end (may exceed sim_end if censored)
    censored: bool
    n_rounds: int
    exits: bool
    n_created: int
    n_recruited: int

    @property
    def n_members(self) -> int:
        return len(self.members)


class _Builder:
    """Mutable state while planning one instance."""

    def __init__(
        self,
        cfg: GeneratorConfig,
        pop: Population,
        fam: Family,
        idx: int,
        fcfg: FamilyCfg,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.pop = pop
        self.fam = fam
        self.idx = idx
        self.fcfg = fcfg
        self.rng = rng
        self.code = FAMILY_CODE[fam]
        self.net = ids.network_id(self.code, idx)
        entity = (self.code << 16) | idx
        self.txn_ids = ids.EventIdAllocator(ids.EVENT_SCENARIO_TXN, entity)
        self.login_ids = ids.EventIdAllocator(ids.EVENT_SCENARIO_LOGIN, entity)
        self.log = EventLog()
        self.sim_end = pop.timebase.sim_end
        self.world = pop.externals[ExternalRole.EXTERNAL_WORLD]
        self.cash = pop.externals[ExternalRole.CASH][0]
        self.last_in: dict[int, int] = {}
        self.intervals: list[tuple[Phase, int, int]] = []
        self.shared_device: tuple[int, DeviceKind] | None = None
        self.shared_ip: int | None = None
        self.created = CreatedEntities()
        self.created_ip_of: dict[int, int] = {}

    # ---- time helpers ----
    def interval(self, phase: Phase, start: int, hours: float) -> int:
        end = start + max(60, int(hours * SECONDS_PER_HOUR))
        self.intervals.append((phase, start, end))
        return len(self.intervals) - 1

    def times_in(self, k: int, n: int) -> list[int]:
        _, s, e = self.intervals[k]
        return sorted(int(x) for x in self.rng.uniform(s + 1, max(s + 2, e - 1), size=n))

    def amount(self, scale: float) -> int:
        return max(
            1,
            int(round(scale * float(self.rng.lognormal(0.0, self.fcfg.instance_params.event_amount_sigma)))),
        )

    # ---- event helpers (every helper returns the transaction event id) ----
    def _keep(self, ts: int) -> bool:
        return ts < self.sim_end

    def inbound(self, dst: Member, amount: int, ts: int, k: int) -> int | None:
        if not self._keep(ts):
            return None
        src = self.world[int(self.rng.integers(len(self.world)))]
        eid = self.log.add_txn(
            event_id=self.txn_ids.next(),
            ts=ts,
            txn_type=TxnType.INBOUND_CREDIT,
            channel=Channel.INBOUND_EXTERNAL,
            src=src,
            dst=dst.account_id,
            amount=amount,
            region=dst.person.region,
            network=self.net,
            phase=self.intervals[k][0],
            event_type=EventType.RECEIVE_FUNDS,
        )
        self.last_in[dst.account_id] = eid
        return eid

    def _login(self, m: Member, ts: int, k: int, channel: Channel) -> tuple[int, int, int]:
        infra = self.cfg.infrastructure
        phase = self.intervals[k][0]
        if self.shared_device is not None and self.shared_ip is not None:
            return emit_session_login(
                self.log,
                self.login_ids,
                self.pop,
                infra,
                m.person,
                m.account_id,
                ts,
                channel,
                0.0,
                self.rng,
                network=self.net,
                phase=phase,
                device=self.shared_device,
                ip=self.shared_ip,
                event_type=EventType.OTHER,
            )
        if m.is_created:
            dev = m.person.devices[0]
            ip = self.created_ip_of[m.account_id]
            return emit_session_login(
                self.log,
                self.login_ids,
                self.pop,
                infra,
                m.person,
                m.account_id,
                ts,
                channel,
                0.0,
                self.rng,
                network=self.net,
                phase=phase,
                device=dev,
                ip=ip,
                event_type=EventType.OTHER,
            )
        return emit_session_login(
            self.log,
            self.login_ids,
            self.pop,
            infra,
            m.person,
            m.account_id,
            ts,
            channel,
            0.0,
            self.rng,
            network=self.net,
            phase=phase,
            event_type=EventType.OTHER,
        )

    def transfer(
        self,
        src: Member,
        dst_account: int,
        amount: int,
        ts: int,
        k: int,
        depends: bool = True,
        event_type: EventType = EventType.TRANSFER_FUNDS,
        external: bool = False,
        dst_member: Member | None = None,
    ) -> int | None:
        if not self._keep(ts):
            return None
        dep = self.last_in.get(src.account_id, -1) if depends else -1
        if depends and dep == -1:
            return None  # the funding inflow was never generated (censored), so neither is this transfer
        channel = Channel.APP if self.rng.random() < 0.7 else Channel.WEB
        lid, dev, ip = self._login(src, ts, k, channel)
        eid = self.log.add_txn(
            event_id=self.txn_ids.next(),
            ts=ts,
            txn_type=TxnType.OUTBOUND_PAYMENT if external else TxnType.TRANSFER,
            channel=channel,
            src=src.account_id,
            dst=dst_account,
            amount=amount,
            region=src.person.region,
            device=dev,
            ip=ip,
            login=lid,
            depends_on=dep,
            network=self.net,
            phase=self.intervals[k][0],
            event_type=event_type,
        )
        if dst_member is not None:
            self.last_in[dst_member.account_id] = eid
        return eid

    def atm_cashout(self, m: Member, total: int, k: int) -> list[int]:
        infra = self.cfg.infrastructure
        step = infra.atm_amount_step_minor
        chunks: list[int] = []
        remaining = total
        while remaining >= step:
            c = min(infra.atm_max_withdrawal_minor, remaining)
            c = (c // step) * step
            chunks.append(c)
            remaining -= c
        out: list[int] = []
        if not chunks:
            return out
        dep = self.last_in.get(m.account_id, -1)
        if dep == -1:
            return out
        pool = self.pop.atms_by_region[m.person.region]
        for ts, c in zip(self.times_in(k, len(chunks)), chunks, strict=True):
            if not self._keep(ts):
                continue
            atm = pool[int(self.rng.integers(len(pool)))]
            out.append(
                self.log.add_txn(
                    event_id=self.txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.ATM_WITHDRAWAL,
                    channel=Channel.ATM,
                    src=m.account_id,
                    dst=self.cash,
                    atm=atm,
                    amount=c,
                    region=m.person.region,
                    depends_on=dep,
                    network=self.net,
                    phase=self.intervals[k][0],
                    event_type=EventType.ATM_WITHDRAWAL,
                )
            )
        return out


def _draw_params(fcfg: FamilyCfg, rng: np.random.Generator, days: int) -> dict[str, float]:
    ip = fcfg.instance_params
    ph = fcfg.phases
    p: dict[str, float] = {
        "amount_scale": float(ip.amount_scale_minor.sample(rng)),
        "fan_degree": float(sample_int(ip.fan_degree, rng, minimum=1)),
        "hops": float(sample_int(ip.hops, rng, minimum=1)),
        "transfers_per_source": float(sample_int(ip.transfers_per_source, rng, minimum=1)),
        "holding_hours": float(ip.holding_hours.sample(rng)),
        "retention": float(np.clip(ip.retention.sample(rng), 0.0, 1.0)),
        "forward_fraction": float(np.clip(ip.forward_fraction.sample(rng), 0.01, 1.0)),
        "start_day": float(np.clip(ip.start_day.sample(rng), 0.0, days)),
        "dormant_gap_days": float(ip.dormant_gap_days.sample(rng)),
        "victim_origin": float(rng.random() < ip.p_victim_origin),
        "setup_hours": float(ph.durations_hours[Phase.SETUP].sample(rng)),
        "movement_hours": float(ph.durations_hours[Phase.MOVEMENT].sample(rng)),
        "exit_hours": float(ph.durations_hours[Phase.EXIT].sample(rng)),
    }
    rounds = 1
    while rounds < ph.max_rounds and rng.random() < ph.p_repeat_round:
        rounds += 1
    p["rounds"] = float(rounds)
    inflow_hours = [float(ph.durations_hours[Phase.INFLOW].sample(rng)) for _ in range(ph.max_rounds)]
    for r in range(ph.max_rounds):
        p[f"inflow_hours_{r}"] = inflow_hours[r]
    p["exits"] = float(rng.random() < ph.p_exit)
    return p


# ---- topology: which member slots a family needs (slot name -> count) ----
def _slots(fam: Family, p: dict[str, float]) -> list[tuple[str, int]]:
    fan = int(p["fan_degree"])
    hops = int(p["hops"])
    victims = fan if p["victim_origin"] else 0
    match fam:
        case Family.FAN_IN:
            return [("victim", victims), ("aggregator", 1), ("cashout", 1)]
        case Family.FAN_OUT:
            return [("distributor", 1), ("receiver", fan)]
        case Family.PASS_THROUGH:
            return [("victim", 1 if p["victim_origin"] else 0), ("relay", 1)]
        case Family.BURST:
            return [("receiver", 1), ("beneficiary", fan)]
        case Family.DORMANT_ACTIVATION:
            return [("dormant", 1), ("receiver", min(2, fan))]
        case Family.MULTI_HOP:
            return [("victim", victims), ("relay", hops), ("cashout", 1)]
        case Family.CYCLE:
            return [("ring", max(3, fan))]
        case Family.STRUCTURING_LIKE:
            return [("victim", victims), ("aggregator", 1)]
        case Family.SHARED_INFRASTRUCTURE:
            return [("member", fan), ("aggregator", 1)]
        case Family.ACCOUNT_TO_CASH:
            return [("victim", 1 if p["victim_origin"] else 0), ("receiver", 1)]
    raise ValueError(f"unknown family {fam}")


def _recruit(
    b: _Builder, part: Partition, slots: list[tuple[str, int]], start_ts: int
) -> tuple[list[Member], int]:
    pop, cfg = b.pop, b.cfg
    need = sum(c for _, c in slots)
    min_age = int(cfg.recruitment.min_account_age_days_at_start * SECONDS_PER_DAY)
    cands = [
        a for a in part.candidates(b.fam, b.idx) if pop.person_of_account(a).opened_at + min_age <= start_ts
    ]
    order = b.rng.permutation(len(cands)) if cands else np.zeros(0, dtype=np.int64)
    picked = [cands[i] for i in order[:need]]
    members: list[Member] = []
    n_created = 0
    it = iter(picked)
    for slot, count in slots:
        for _ in range(count):
            acc = next(it, None)
            if acc is not None:
                members.append(Member(acc, pop.person_of_account(acc), False, slot))
                continue
            # Shortfall: open a scenario-namespace account (recorded, never silent).
            payload = ids.scenario_payload(b.code, b.idx, n_created)
            region = int(b.rng.integers(pop.n_regions))
            opened = start_ts - int(b.rng.uniform(1, 14) * SECONDS_PER_DAY)
            dev = ids.pack(ids.DEVICE_SCENARIO, payload)
            ip = ids.pack(ids.IP_SCENARIO, payload)
            person = Person(
                pid=ids.pack(ids.PERSON_SCENARIO, payload),
                archetype=None,  # type: ignore[arg-type]
                region=region,
                household=-1,
                account_id=ids.pack(ids.ACCOUNT_SCENARIO, payload),
                opened_at=opened,
                overdraft_limit=0,
                initial_balance=0,
                devices=[(dev, DeviceKind.MOBILE)],
            )
            b.created.persons.append(person)
            b.created.devices[dev] = (DeviceKind.MOBILE, opened)
            b.created.device_controllers[dev] = [person.pid]
            b.created.ips[ip] = (IpContext.RESIDENTIAL_SINGLE, region)
            b.created_ip_of[person.account_id] = ip
            b.created.uses_ip.append((dev, ip))
            members.append(Member(person.account_id, person, True, slot))
            n_created += 1
    return members, n_created


def plan_instance(
    cfg: GeneratorConfig, pop: Population, part: Partition, fam: Family, idx: int
) -> InstancePlan:
    fcfg = cfg.scenarios[fam]
    rng = stream(cfg.simulation.seed, "scenario", fam.value, idx)
    tb = pop.timebase
    p = _draw_params(fcfg, rng, tb.days)
    b = _Builder(cfg, pop, fam, idx, fcfg, rng)
    start_ts = max(int(p["start_day"] * SECONDS_PER_DAY), START_MARGIN_S + 3600)
    slots = _slots(fam, p)
    members, n_created = _recruit(b, part, slots, start_ts)
    by_slot: dict[str, list[Member]] = {}
    for m in members:
        by_slot.setdefault(m.slot, []).append(m)

    def S(name: str) -> list[Member]:
        return by_slot.get(name, [])

    scale = p["amount_scale"]
    frac = p["forward_fraction"]
    hold_h = p["holding_hours"]
    rounds = int(p["rounds"])
    exits = bool(p["exits"])
    role_log: list[tuple[Member, int, Role]] = []

    def role(m: Member, k: int, r: Role) -> None:
        role_log.append((m, k, r))

    planned: dict[int, float] = {}

    def add_planned(m: Member, amt: int) -> None:
        planned[m.account_id] = planned.get(m.account_id, 0.0) + amt

    t = start_ts
    # SETUP (shared infrastructure: members log in from a scenario-controlled device and IP)
    if fam == Family.SHARED_INFRASTRUCTURE:
        k = b.interval(Phase.SETUP, t, p["setup_hours"])
        payload = ids.scenario_payload(b.code, idx, 0xFFFF)
        dev = ids.pack(ids.DEVICE_SCENARIO, payload)
        ip = ids.pack(ids.IP_SCENARIO, payload)
        b.created.devices[dev] = (DeviceKind.MOBILE, t)
        b.created.device_controllers[dev] = []
        b.created.ips[ip] = (IpContext.RESIDENTIAL_SINGLE, int(rng.integers(pop.n_regions)))
        b.shared_device, b.shared_ip = (dev, DeviceKind.MOBILE), ip
        b.created.uses_ip.append((dev, ip))
        for m, ts in zip(S("member") + S("aggregator"), b.times_in(k, len(S("member")) + 1), strict=True):
            if ts < b.sim_end:
                b.log.add_login(
                    event_id=b.login_ids.next(),
                    ts=ts,
                    account=m.account_id,
                    device=dev,
                    ip=ip,
                    channel=Channel.APP,
                    outcome=LoginOutcome.SUCCESS,
                    network=b.net,
                    phase=Phase.SETUP,
                    event_type=EventType.NEW_DEVICE_LOGIN,
                )
            role(m, k, Role.SUSPICIOUS_UNKNOWN)
        t = b.intervals[k][2]

    # (INFLOW -> HOLD) x rounds
    first_receivers: list[Member]
    match fam:
        case Family.FAN_IN | Family.STRUCTURING_LIKE:
            first_receivers = S("aggregator")
        case Family.FAN_OUT:
            first_receivers = S("distributor")
        case Family.PASS_THROUGH:
            first_receivers = S("relay")
        case Family.BURST | Family.ACCOUNT_TO_CASH:
            first_receivers = S("receiver")
        case Family.DORMANT_ACTIVATION:
            first_receivers = S("dormant")
        case Family.MULTI_HOP:
            first_receivers = S("relay")[:1]
        case Family.CYCLE:
            first_receivers = S("ring")[:1]
        case Family.SHARED_INFRASTRUCTURE:
            first_receivers = S("member")
    receiver_role = {Family.FAN_IN: Role.AGGREGATOR, Family.STRUCTURING_LIKE: Role.AGGREGATOR}.get(
        fam, Role.FUND_RECEIVER
    )
    band = fcfg.instance_params.amount_band_width

    for r in range(rounds):
        k = b.interval(Phase.INFLOW, t, p[f"inflow_hours_{r}"])
        victims = S("victim")
        for rcv in first_receivers:
            role(rcv, k, receiver_role)
        if victims:
            per_source = int(p["transfers_per_source"])
            for v in victims:
                role(v, k, Role.VICTIM_LIKE)
                dst = first_receivers[0]
                for ts in b.times_in(k, per_source):
                    if fam == Family.STRUCTURING_LIKE:
                        amt = max(1, int(round(scale * rng.uniform(1.0 - band, 1.0))))
                    else:
                        amt = b.amount(scale)
                    if b.transfer(v, dst.account_id, amt, ts, k, depends=False, dst_member=dst) is not None:
                        add_planned(dst, amt)
        else:
            n_sources = {
                Family.FAN_IN: int(p["fan_degree"]),
                Family.STRUCTURING_LIKE: int(p["fan_degree"]),
            }.get(fam, 1)
            per_source = int(p["transfers_per_source"]) if fam == Family.STRUCTURING_LIKE else 1
            for rcv in first_receivers:
                mult = p["fan_degree"] if fam in (Family.FAN_OUT, Family.BURST) else 1.0
                for ts in b.times_in(k, n_sources * per_source):
                    if fam == Family.STRUCTURING_LIKE:
                        amt = max(1, int(round(scale * rng.uniform(1.0 - band, 1.0))))
                    else:
                        amt = b.amount(scale * mult)
                    if b.inbound(rcv, amt, ts, k) is not None:
                        add_planned(rcv, amt)
        t = b.intervals[k][2]
        k = b.interval(Phase.HOLD, t, hold_h)
        for rcv in first_receivers:
            role(rcv, k, receiver_role)
        t = b.intervals[k][2]

    # MOVEMENT
    exit_holders: list[Member] = []
    moved: dict[int, float] = {}

    def move(
        src: Member, dst: Member, amount: float, ts: int, k: int, et: EventType = EventType.TRANSFER_FUNDS
    ) -> None:
        amt = int(round(amount))
        if amt <= 0:
            return
        if b.transfer(src, dst.account_id, amt, ts, k, dst_member=dst, event_type=et) is not None:
            moved[dst.account_id] = moved.get(dst.account_id, 0.0) + amt

    k_mv = None
    match fam:
        case Family.FAN_IN:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            agg, co = S("aggregator")[0], S("cashout")[0]
            role(agg, k_mv, Role.RELAY)
            role(co, k_mv, Role.FUND_RECEIVER)
            move(agg, co, planned.get(agg.account_id, 0.0) * frac, b.times_in(k_mv, 1)[0], k_mv)
            exit_holders = [co]
        case Family.FAN_OUT | Family.BURST:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            src = first_receivers[0]
            outs = S("receiver") if fam == Family.FAN_OUT else S("beneficiary")
            role(src, k_mv, Role.DISTRIBUTOR)
            share = planned.get(src.account_id, 0.0) * frac / max(1, len(outs))
            for m, ts in zip(outs, b.times_in(k_mv, len(outs)), strict=True):
                role(m, k_mv, Role.FUND_RECEIVER)
                move(src, m, share * rng.uniform(0.85, 1.15), ts, k_mv, EventType.MULTI_TRANSFER)
            exit_holders = outs
        case Family.DORMANT_ACTIVATION:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            src = S("dormant")[0]
            outs = S("receiver")
            role(src, k_mv, Role.DISTRIBUTOR if len(outs) > 1 else Role.RELAY)
            share = planned.get(src.account_id, 0.0) * frac / max(1, len(outs))
            for m, ts in zip(outs, b.times_in(k_mv, len(outs)), strict=True):
                role(m, k_mv, Role.FUND_RECEIVER)
                move(src, m, share, ts, k_mv)
            exit_holders = outs
        case Family.MULTI_HOP:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            chain = S("relay") + S("cashout")
            _, s0, e0 = b.intervals[k_mv]
            step = max(2, (e0 - s0) // (len(chain) + 1))
            amount = planned.get(chain[0].account_id, 0.0)
            for j in range(len(chain) - 1):
                amount *= frac
                role(chain[j], k_mv, Role.RELAY)
                ts = s0 + (j + 1) * step + int(rng.integers(0, max(1, step // 2)))
                move(chain[j], chain[j + 1], amount, ts, k_mv)
            role(chain[-1], k_mv, Role.FUND_RECEIVER)
            exit_holders = [chain[-1]]
        case Family.CYCLE:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            ring = S("ring")
            loops = int(p["hops"])
            n_steps = loops * len(ring)
            _, s0, e0 = b.intervals[k_mv]
            step = max(2, (e0 - s0) // (n_steps + 1))
            amount = planned.get(ring[0].account_id, 0.0)
            for j in range(n_steps):
                a_, c_ = ring[j % len(ring)], ring[(j + 1) % len(ring)]
                amount *= frac
                role(a_, k_mv, Role.RELAY)
                move(a_, c_, amount, s0 + (j + 1) * step, k_mv)
            exit_holders = [ring[0]]
        case Family.SHARED_INFRASTRUCTURE:
            k_mv = b.interval(Phase.MOVEMENT, t, p["movement_hours"])
            agg = S("aggregator")[0]
            role(agg, k_mv, Role.AGGREGATOR)
            for m, ts in zip(S("member"), b.times_in(k_mv, len(S("member"))), strict=True):
                role(m, k_mv, Role.RELAY)
                move(m, agg, planned.get(m.account_id, 0.0) * frac, ts, k_mv)
            exit_holders = [agg]
        case Family.PASS_THROUGH | Family.ACCOUNT_TO_CASH | Family.STRUCTURING_LIKE:
            exit_holders = list(first_receivers)
            for m in exit_holders:
                moved[m.account_id] = planned.get(m.account_id, 0.0)
    if k_mv is not None:
        t = b.intervals[k_mv][2]

    # EXIT (terminal high-risk phase) or INACTIVE (network abandoned without exit)
    if exits:
        k = b.interval(Phase.EXIT, t, p["exit_hours"])
        external_exit = fam in (Family.PASS_THROUGH, Family.CYCLE, Family.SHARED_INFRASTRUCTURE)
        for m in exit_holders:
            held = moved.get(m.account_id, 0.0) * frac
            if held < 1:
                continue
            if external_exit:
                role(m, k, Role.RELAY)
                ext_dst = b.world[int(rng.integers(len(b.world)))]
                b.transfer(m, ext_dst, int(round(held)), b.times_in(k, 1)[0], k, external=True)
            else:
                role(m, k, Role.CASH_OUT_RISK)
                b.atm_cashout(m, int(round(held)), k)
        t = b.intervals[k][2]
    end_ts = t

    # Member suppression windows (retention of normal activity while involved; dormant: none for a long time).
    suppressions: list[Suppression] = []
    for m in members:
        if m.is_created or m.slot == "victim":
            continue
        if m.slot == "dormant":
            gap = int(p["dormant_gap_days"] * SECONDS_PER_DAY)
            suppressions.append(Suppression(m.account_id, start_ts - gap, end_ts, 0.0))
        else:
            suppressions.append(Suppression(m.account_id, start_ts, end_ts, p["retention"]))

    # Roles per member per interval; joined_ts = start of first interval with a role.
    for m, k, rl in role_log:
        m.roles[k] = rl
        s = b.intervals[k][1]
        m.joined_ts = s if m.joined_ts is None else min(m.joined_ts, s)
    clipped = [(ph, s, min(e, b.sim_end)) for ph, s, e in b.intervals]
    return InstancePlan(
        family=fam,
        instance_idx=idx,
        network_id=b.net,
        scenario_id=f"{fam.value}:{idx}",
        is_ood_family=fcfg.is_ood_family,
        params=p,
        members=members,
        intervals=clipped,
        log=b.log,
        created=b.created,
        suppressions=suppressions,
        start_ts=start_ts,
        end_ts=end_ts,
        censored=end_ts > b.sim_end,
        n_rounds=rounds,
        exits=exits,
        n_created=n_created,
        n_recruited=len(members) - n_created,
    )


PlanFn = Callable[[GeneratorConfig, Population, Partition, Family, int], InstancePlan]
