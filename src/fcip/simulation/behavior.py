"""Normal behavior engine: one generic, config-driven engine for all archetypes.

Each person's behavior uses only its own stream ``("normal", archetype, person_id)`` and its own event-ID
namespace, so it does not depend on any scenario. All plan-level draws (travel, surge, dormancy windows,
personal rates) happen first and in a fixed order; the dormancy gap is applied afterwards as a filter that
consumes no randomness, so kept events are identical whether or not a gap exists.
"""

from __future__ import annotations

import numpy as np

from fcip.common import ids
from fcip.common.rng import stream
from fcip.common.taxonomy import Channel, ExternalRole, LoginOutcome, TxnType
from fcip.common.timebase import SECONDS_PER_DAY, SECONDS_PER_HOUR
from fcip.config.distributions import sample_int
from fcip.config.generator import ArchetypeCfg, GeneratorConfig
from fcip.simulation.events import EventLog
from fcip.simulation.logins import choose_device, choose_ip, emit_session_login
from fcip.simulation.population import Person, Population

# Earliest event time after an account opens (leaves room for the preceding login session).
START_MARGIN_S = 2 * SECONDS_PER_HOUR


def _uniform_times(rng: np.random.Generator, days: np.ndarray, h_lo: float, h_hi: float) -> np.ndarray:
    secs = rng.uniform(h_lo * SECONDS_PER_HOUR, h_hi * SECONDS_PER_HOUR, size=len(days))
    return days.astype(np.int64) * SECONDS_PER_DAY + secs.astype(np.int64)


class _Windows:
    """Per-person time windows drawn up front: travel trips, one surge, one dormancy gap."""

    def __init__(self) -> None:
        self.trips: list[tuple[int, int, int]] = []
        self.surge: tuple[int, int, float, float] | None = None
        self.gap: tuple[int, int] | None = None

    def travel_region(self, ts: int) -> int | None:
        for s, e, r in self.trips:
            if s <= ts < e:
                return r
        return None

    def surge_factors(self, ts: int) -> tuple[float, float]:
        if self.surge is not None and self.surge[0] <= ts < self.surge[1]:
            return self.surge[2], self.surge[3]
        return 1.0, 1.0


def _draw_windows(
    a: ArchetypeCfg, p: Person, pop: Population, rng: np.random.Generator, start: int, end: int
) -> _Windows:
    w = _Windows()
    span_days = max(1, (end - start) // SECONDS_PER_DAY)
    if a.travel is not None:
        for _ in range(sample_int(a.travel.trips, rng)):
            d0 = start + int(rng.integers(span_days)) * SECONDS_PER_DAY
            length = max(1, int(round(float(a.travel.trip_days.sample(rng))))) * SECONDS_PER_DAY
            others = [r for r in range(pop.n_regions) if r != p.region] or [p.region]
            w.trips.append((d0, d0 + length, others[int(rng.integers(len(others)))]))
    if a.surge is not None:
        d0 = start + int(rng.integers(span_days)) * SECONDS_PER_DAY
        length = max(1, int(round(float(a.surge.duration_days.sample(rng))))) * SECONDS_PER_DAY
        w.surge = (
            d0,
            d0 + length,
            float(a.surge.rate_multiplier.sample(rng)),
            float(a.surge.amount_multiplier.sample(rng)),
        )
    if a.dormancy is not None:
        has_gap = rng.random() < a.dormancy.p_dormant_gap
        length = max(1, int(round(float(a.dormancy.gap_days.sample(rng))))) * SECONDS_PER_DAY
        d0 = start + int(rng.integers(span_days)) * SECONDS_PER_DAY
        if has_gap:
            w.gap = (d0, d0 + length)
    return w


def _counterparty_open(pop: Population, account_id: int, ts: int) -> bool:
    """External accounts are always open; internal ones only from their opening time (plus margin)."""
    pid = pop.account_to_pid.get(account_id)
    return pid is None or pop.persons[pid].opened_at + START_MARGIN_S <= ts


def _open_account_of(pop: Population, pid: int, ts: int) -> int | None:
    q = pop.persons[pid]
    return q.account_id if q.opened_at + START_MARGIN_S <= ts else None


def generate_person(pop: Population, cfg: GeneratorConfig, p: Person) -> EventLog:
    seed = cfg.simulation.seed
    infra = cfg.infrastructure
    a = cfg.population.archetypes[p.archetype]
    rng = stream(seed, "normal", p.archetype.value, p.pid)
    txn_ids = ids.EventIdAllocator(ids.EVENT_NORMAL_TXN, p.pid)
    login_ids = ids.EventIdAllocator(ids.EVENT_NORMAL_LOGIN, p.pid)
    log = EventLog()
    tb = pop.timebase
    sim_end = tb.sim_end
    start = max(p.opened_at, 0) + START_MARGIN_S
    end = sim_end - SECONDS_PER_HOUR
    acct = p.account_id
    if start >= end:
        return log
    w = _draw_windows(a, p, pop, rng, start, end)
    first_day = start // SECONDS_PER_DAY + (1 if start % SECONDS_PER_DAY else 0)
    days = np.arange(first_day, tb.days, dtype=np.int64)
    p_fail = a.logins.p_failure

    def digital(ts: int, channel: Channel, account: int, person: Person) -> tuple[int, int, int]:
        return emit_session_login(
            log,
            login_ids,
            pop,
            infra,
            person,
            account,
            ts,
            channel,
            p_fail,
            rng,
            travel_region=w.travel_region(ts),
        )

    # 1. income (monthly)
    if a.income is not None and p.employer_account is not None:
        inc_days = tb.monthly_day_indices(a.income.day_of_month)
        inc_days = inc_days[inc_days >= first_day]
        base_amount = float(a.income.amount.sample(rng))
        for ts in _uniform_times(rng, inc_days, 6, 8):
            amount = int(round(base_amount * rng.uniform(0.97, 1.03)))
            if not _counterparty_open(pop, p.employer_account, int(ts)):
                continue
            if p.employer_is_internal:
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.TRANSFER,
                    channel=Channel.SCHEDULED,
                    src=p.employer_account,
                    dst=acct,
                    amount=amount,
                    region=p.region,
                )
            else:
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.INBOUND_CREDIT,
                    channel=Channel.INBOUND_EXTERNAL,
                    src=p.employer_account,
                    dst=acct,
                    amount=amount,
                    region=p.region,
                )

    # 2. rent (monthly, standing order)
    if a.rent is not None and p.landlord_account is not None:
        rent_days = tb.monthly_day_indices(a.rent.day_of_month)
        rent_days = rent_days[rent_days >= first_day]
        amount = max(1, int(round(float(a.rent.amount.sample(rng)))))
        ttype = TxnType.TRANSFER if p.landlord_is_internal else TxnType.OUTBOUND_PAYMENT
        for ts in _uniform_times(rng, rent_days, 5, 7):
            if not _counterparty_open(pop, p.landlord_account, int(ts)):
                continue
            log.add_txn(
                event_id=txn_ids.next(),
                ts=ts,
                txn_type=ttype,
                channel=Channel.SCHEDULED,
                src=acct,
                dst=p.landlord_account,
                amount=amount,
                region=p.region,
            )

    # 3. bills (monthly direct debits)
    if a.bills is not None:
        billers = pop.externals[ExternalRole.BILLER]
        for _ in range(sample_int(a.bills.n_bills, rng)):
            biller = billers[int(rng.integers(len(billers)))]
            dom = int(rng.integers(1, 29))
            base = float(a.bills.amount.sample(rng))
            bill_days = tb.monthly_day_indices(dom)
            bill_days = bill_days[bill_days >= first_day]
            for ts in _uniform_times(rng, bill_days, 3, 6):
                amount = max(1, int(round(base * rng.uniform(0.9, 1.1))))
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.OUTBOUND_PAYMENT,
                    channel=Channel.SCHEDULED,
                    src=acct,
                    dst=biller,
                    amount=amount,
                    region=p.region,
                )

    # 4. card payments at merchants (POS)
    if a.card is not None and p.favorite_merchants:
        card_net = pop.externals[ExternalRole.CARD_NETWORK][0]
        rate = float(a.card.daily_rate.sample(rng))
        for d in days:
            day_ts = int(d) * SECONDS_PER_DAY
            rate_mult, amt_mult = w.surge_factors(day_ts + 12 * SECONDS_PER_HOUR)
            for _ in range(int(rng.poisson(rate * rate_mult))):
                if rng.random() < a.card.p_night:
                    ts = day_ts + int(rng.uniform(0, 5 * SECONDS_PER_HOUR))
                else:
                    ts = day_ts + int(rng.uniform(8 * SECONDS_PER_HOUR, 22 * SECONDS_PER_HOUR))
                trip = w.travel_region(ts)
                if trip is None:
                    merchant = p.favorite_merchants[int(rng.integers(len(p.favorite_merchants)))]
                else:
                    pool = pop.merchants_by_region[trip]
                    merchant = pool[int(rng.integers(len(pool)))]
                amount = max(1, int(round(float(a.card.amount.sample(rng)) * amt_mult)))
                if start <= ts < end:
                    log.add_txn(
                        event_id=txn_ids.next(),
                        ts=ts,
                        txn_type=TxnType.CARD_PAYMENT,
                        channel=Channel.POS,
                        src=acct,
                        dst=card_net,
                        merchant=merchant,
                        amount=amount,
                        region=pop.merchants[merchant][1],
                    )

    # 5. ATM withdrawals
    if a.atm is not None:
        cash = pop.externals[ExternalRole.CASH][0]
        step = infra.atm_amount_step_minor
        rate = float(a.atm.weekly_rate.sample(rng)) / 7.0
        for d in days:
            for _ in range(int(rng.poisson(rate))):
                ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(7 * SECONDS_PER_HOUR, 23 * SECONDS_PER_HOUR))
                region = w.travel_region(ts)
                region = p.region if region is None else region
                pool = pop.atms_by_region[region]
                atm = pool[int(rng.integers(len(pool)))]
                raw = float(a.atm.amount.sample(rng))
                amount = int(min(infra.atm_max_withdrawal_minor, max(step, round(raw / step) * step)))
                if start <= ts < end:
                    log.add_txn(
                        event_id=txn_ids.next(),
                        ts=ts,
                        txn_type=TxnType.ATM_WITHDRAWAL,
                        channel=Channel.ATM,
                        src=acct,
                        dst=cash,
                        atm=atm,
                        amount=amount,
                        region=region,
                    )

    # 6. P2P transfers to contacts (digital, with login)
    if a.p2p is not None and p.contacts:
        rate = float(a.p2p.monthly_rate.sample(rng)) / 30.0
        for d in days:
            for _ in range(int(rng.poisson(rate))):
                ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(8 * SECONDS_PER_HOUR, 23 * SECONDS_PER_HOUR))
                contact = p.contacts[int(rng.integers(len(p.contacts)))]
                amount = max(1, int(round(float(a.p2p.amount.sample(rng)))))
                channel = Channel.WEB if rng.random() < a.p2p.p_web else Channel.APP
                dst = _open_account_of(pop, contact, ts)
                if dst is None or not (start <= ts < end):
                    continue
                lid, dev, ip = digital(ts, channel, acct, p)
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.TRANSFER,
                    channel=channel,
                    src=acct,
                    dst=dst,
                    amount=amount,
                    region=p.region,
                    device=dev,
                    ip=ip,
                    login=lid,
                )

    # 7. business flows: customer payments in, supplier payments out, internal payouts out
    if a.business is not None:
        b = a.business
        world = pop.externals[ExternalRole.EXTERNAL_WORLD]
        n_persons = len(pop.persons)
        cust_rate = float(b.daily_customer_payments.sample(rng))
        supp_rate = float(b.weekly_supplier_payments.sample(rng)) / 7.0
        pay_rate = float(b.daily_internal_payouts.sample(rng))
        acquirer = world[int(rng.integers(len(world)))]
        for d in days:
            batch = 0  # external (card-acquired) customer revenue settles as one credit per day
            for _ in range(int(rng.poisson(cust_rate))):
                ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(8 * SECONDS_PER_HOUR, 21 * SECONDS_PER_HOUR))
                amount = max(1, int(round(float(b.customer_amount.sample(rng)))))
                if rng.random() < b.p_internal_customer:
                    cid = int(rng.integers(n_persons))
                    src = _open_account_of(pop, cid, ts)
                    if cid == p.pid or src is None or not (start <= ts < end):
                        continue
                    customer = pop.persons[cid]
                    lid, dev, ip = emit_session_login(
                        log, login_ids, pop, infra, customer, src, ts, Channel.APP, p_fail, rng
                    )
                    log.add_txn(
                        event_id=txn_ids.next(),
                        ts=ts,
                        txn_type=TxnType.TRANSFER,
                        channel=Channel.APP,
                        src=src,
                        dst=acct,
                        amount=amount,
                        region=customer.region,
                        device=dev,
                        ip=ip,
                        login=lid,
                    )
                else:
                    batch += amount
            settle_ts = int(d) * SECONDS_PER_DAY + 23 * SECONDS_PER_HOUR + int(rng.integers(0, 1800))
            if batch > 0 and start <= settle_ts < end:
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=settle_ts,
                    txn_type=TxnType.INBOUND_CREDIT,
                    channel=Channel.INBOUND_EXTERNAL,
                    src=acquirer,
                    dst=acct,
                    amount=batch,
                    region=p.region,
                )
            for _ in range(int(rng.poisson(supp_rate))):
                ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(9 * SECONDS_PER_HOUR, 18 * SECONDS_PER_HOUR))
                amount = max(1, int(round(float(b.supplier_amount.sample(rng)))))
                dst = world[int(rng.integers(len(world)))]
                if not (start <= ts < end):
                    continue
                lid, dev, ip = digital(ts, Channel.WEB, acct, p)
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.OUTBOUND_PAYMENT,
                    channel=Channel.WEB,
                    src=acct,
                    dst=dst,
                    amount=amount,
                    region=p.region,
                    device=dev,
                    ip=ip,
                    login=lid,
                )
            for _ in range(int(rng.poisson(pay_rate))):
                ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(9 * SECONDS_PER_HOUR, 18 * SECONDS_PER_HOUR))
                amount = max(1, int(round(float(b.payout_amount.sample(rng)))))
                rid = int(rng.integers(n_persons))
                dst = _open_account_of(pop, rid, ts)
                if rid == p.pid or dst is None or not (start <= ts < end):
                    continue
                lid, dev, ip = digital(ts, Channel.WEB, acct, p)
                log.add_txn(
                    event_id=txn_ids.next(),
                    ts=ts,
                    txn_type=TxnType.TRANSFER,
                    channel=Channel.WEB,
                    src=acct,
                    dst=dst,
                    amount=amount,
                    region=p.region,
                    device=dev,
                    ip=ip,
                    login=lid,
                )

    # 8. browsing logins without a transaction
    rate = float(a.logins.daily_browse_rate.sample(rng))
    for d in days:
        for _ in range(int(rng.poisson(rate))):
            ts = int(d) * SECONDS_PER_DAY + int(rng.uniform(7 * SECONDS_PER_HOUR, 23.5 * SECONDS_PER_HOUR))
            channel = Channel.APP if rng.random() < 0.7 else Channel.WEB
            bdev = choose_device(p, channel, rng)
            ip = choose_ip(pop, infra, p, bdev[1], ts, rng, w.travel_region(ts))
            if start <= ts < end:
                log.add_login(
                    event_id=login_ids.next(),
                    ts=ts,
                    account=acct,
                    device=bdev[0],
                    ip=ip,
                    channel=channel,
                    outcome=LoginOutcome.SUCCESS,
                )

    # Dormancy gap (confounder: legitimately inactive period): drop this stream's events inside the gap.
    if w.gap is not None:
        g0, g1 = w.gap
        keep_t = (
            ~((np.asarray(log.t_ts) >= g0) & (np.asarray(log.t_ts) < g1)) if log.n_txn else np.zeros(0, bool)
        )
        dropped_logins = {log.t_login[i] for i in np.flatnonzero(~keep_t) if log.t_login[i] is not None}
        lts = np.asarray(log.l_ts)
        keep_l = ~((lts >= g0) & (lts < g1)) if log.n_login else np.zeros(0, bool)
        if log.n_login:
            keep_l &= np.asarray([lid not in dropped_logins for lid in log.l_event_id])
            # a login kept outside the gap may belong to a transaction inside it: drop it too
            kept_txn_logins = {log.t_login[i] for i in np.flatnonzero(keep_t) if log.t_login[i] is not None}
            txn_logins = {x for x in log.t_login if x is not None}
            keep_l &= np.asarray(
                [(lid not in txn_logins) or (lid in kept_txn_logins) for lid in log.l_event_id]
            )
        log = log.filter(keep_t, keep_l)
    return log
