"""Login context: which device and IP a person uses at a given time, and login-event emission.

Only APP and WEB transactions have a login step (decision 0005 channel table). Shared IPs (household,
corporate NAT, carrier NAT, public Wi-Fi) are ordinary here, so a shared IP alone carries no suspicion.
"""

from __future__ import annotations

import numpy as np

from fcip.common.ids import EventIdAllocator
from fcip.common.taxonomy import Channel, DeviceKind, EventType, LoginOutcome, Phase
from fcip.common.timebase import TimeBase
from fcip.config.generator import InfrastructureCfg
from fcip.simulation.events import NO_NETWORK, EventLog
from fcip.simulation.population import Person, Population


def choose_device(person: Person, channel: Channel, rng: np.random.Generator) -> tuple[int, DeviceKind]:
    if channel == Channel.APP:
        pool = [d for d in person.devices if d[1] in (DeviceKind.MOBILE, DeviceKind.TABLET)]
    else:
        pool = [d for d in person.devices if d[1] == DeviceKind.DESKTOP] or list(person.devices)
    if not pool:
        raise ValueError(f"person {person.pid} has no device usable for {channel}")
    return pool[int(rng.integers(len(pool)))]


def choose_ip(
    pop: Population,
    infra: InfrastructureCfg,
    person: Person,
    kind: DeviceKind,
    ts: int,
    rng: np.random.Generator,
    travel_region: int | None = None,
) -> int:
    if travel_region is not None:
        if rng.random() < 0.5:
            pool = pop.public_wifi_by_region[travel_region]
        else:
            pool = pop.cgnat_by_region[travel_region]
        return pool[int(rng.integers(len(pool)))]
    hour = int(TimeBase.hour_of_day(ts))
    weekday = int(TimeBase.weekday(ts))
    work_hours = weekday < 5 and 9 <= hour < 17
    home_ip = pop.households[person.household]["ip"]
    if kind == DeviceKind.TABLET:
        return home_ip
    if kind == DeviceKind.DESKTOP:
        return person.workplace_ip if (work_hours and person.workplace_ip is not None) else home_ip
    if rng.random() < infra.p_public_wifi:
        pool = pop.public_wifi_by_region[person.region]
        return pool[int(rng.integers(len(pool)))]
    if not work_hours and rng.random() < infra.p_home_wifi_mobile:
        return home_ip
    if work_hours and person.workplace_ip is not None and rng.random() < 0.5:
        return person.workplace_ip
    pool = pop.cgnat_by_region[person.region]
    return pool[int(rng.integers(len(pool)))]


def emit_session_login(
    log: EventLog,
    alloc: EventIdAllocator,
    pop: Population,
    infra: InfrastructureCfg,
    person: Person,
    account: int,
    txn_ts: int,
    channel: Channel,
    p_failure: float,
    rng: np.random.Generator,
    travel_region: int | None = None,
    network: int = NO_NETWORK,
    phase: Phase | None = None,
    device: tuple[int, DeviceKind] | None = None,
    ip: int | None = None,
    event_type: EventType | None = None,
) -> tuple[int, int, int]:
    """Emit a successful login shortly before ``txn_ts`` (optionally preceded by a failed attempt).

    Returns ``(login_event_id, device_id, ip_id)`` for the transaction to reference.
    """
    if channel not in (Channel.APP, Channel.WEB):
        raise ValueError(f"no login step for channel {channel}")
    dev = device if device is not None else choose_device(person, channel, rng)
    ip_id = ip if ip is not None else choose_ip(pop, infra, person, dev[1], txn_ts, rng, travel_region)
    session = int(max(5.0, float(infra.session_seconds.sample(rng))))
    login_ts = txn_ts - session
    fail = rng.random() < p_failure
    if fail:
        log.add_login(
            event_id=alloc.next(),
            ts=login_ts - int(rng.integers(5, 60)),
            account=account,
            device=dev[0],
            ip=ip_id,
            channel=channel,
            outcome=LoginOutcome.FAILURE,
            network=network,
            phase=phase,
            event_type=event_type,
        )
    lid = log.add_login(
        event_id=alloc.next(),
        ts=login_ts,
        account=account,
        device=dev[0],
        ip=ip_id,
        channel=channel,
        outcome=LoginOutcome.SUCCESS,
        network=network,
        phase=phase,
        event_type=event_type,
    )
    return lid, dev[0], ip_id
