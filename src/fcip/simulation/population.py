"""Population and infrastructure: persons, households, accounts, devices, IPs, ATMs, merchants, external
accounts, and the static social / economic links used by normal behavior.

Every random draw here uses a ``("population", <table>)`` stream (decision 0002), so the population is a
pure function of ``(seed, config)`` and does not depend on which scenarios are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fcip.common import ids
from fcip.common.rng import stream
from fcip.common.taxonomy import Archetype, DeviceKind, ExternalRole, IpContext
from fcip.common.timebase import SECONDS_PER_DAY, TimeBase
from fcip.config.distributions import sample_int
from fcip.config.generator import BusinessCfg, GeneratorConfig


@dataclass
class Person:
    pid: int
    archetype: Archetype
    region: int
    household: int
    account_id: int
    opened_at: int
    overdraft_limit: int
    initial_balance: int
    devices: list[tuple[int, DeviceKind]] = field(default_factory=list)
    workplace_ip: int | None = None
    contacts: list[int] = field(default_factory=list)
    employer_account: int | None = None
    employer_is_internal: bool = False
    landlord_account: int | None = None
    landlord_is_internal: bool = False
    favorite_merchants: list[int] = field(default_factory=list)


@dataclass
class ExternalAccount:
    account_id: int
    role: ExternalRole


@dataclass
class Population:
    timebase: TimeBase
    persons: list[Person]
    households: dict[int, dict]  # hh id -> {"region", "ip", "shared_device" | None, "members"}
    externals: dict[ExternalRole, list[int]]
    external_accounts: list[ExternalAccount]
    devices: dict[int, tuple[DeviceKind, int]]  # device_id -> (kind, first_seen_at)
    ips: dict[int, tuple[IpContext, int]]  # ip_id -> (context, region)
    atms: dict[int, int]  # atm_id -> region
    merchants: dict[int, tuple[str, int]]  # merchant_id -> (category, region)
    atms_by_region: dict[int, list[int]]
    merchants_by_region: dict[int, list[int]]
    public_wifi_by_region: dict[int, list[int]]
    corporate_by_region: dict[int, list[int]]
    cgnat_by_region: dict[int, list[int]]
    business_pids: list[int]  # persons whose archetype has a business block
    account_to_pid: dict[int, int]
    device_controllers: dict[int, list[int]]  # device_id -> controlling person ids

    @property
    def n_regions(self) -> int:
        return len(self.atms_by_region)

    def person_of_account(self, account_id: int) -> Person:
        return self.persons[self.account_to_pid[account_id]]


def _choose_archetypes(cfg: GeneratorConfig, rng: np.random.Generator) -> list[Archetype]:
    names = list(cfg.population.archetype_mix)
    probs = np.asarray([cfg.population.archetype_mix[a] for a in names], dtype=np.float64)
    idx = rng.choice(len(names), size=cfg.population.n_persons, p=probs / probs.sum())
    return [names[i] for i in idx]


def _business(cfg: GeneratorConfig, archetype: Archetype) -> BusinessCfg:
    b = cfg.population.archetypes[archetype].business
    if b is None:
        raise ValueError(f"archetype {archetype} has no business block")
    return b


def build_population(cfg: GeneratorConfig) -> Population:
    seed = cfg.simulation.seed
    tb = TimeBase(cfg.simulation.days, cfg.simulation.calendar_anchor)
    infra = cfg.infrastructure
    n = cfg.population.n_persons
    n_regions = infra.n_regions

    # ---- infrastructure: ATMs, merchants, IP pools ----
    rng = stream(seed, "population", "atms")
    atms: dict[int, int] = {}
    atms_by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    for r in range(n_regions):
        for k in range(infra.n_atms_per_region):
            aid = r * 1000 + k
            atms[aid] = r
            atms_by_region[r].append(aid)
    del rng  # ATMs are fully determined by the config; stream kept for symmetry of the path registry

    rng = stream(seed, "population", "merchants")
    cats = list(infra.merchant_category_mix)
    cat_p = np.asarray([infra.merchant_category_mix[c] for c in cats])
    merchants: dict[int, tuple[str, int]] = {}
    merchants_by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    for r in range(n_regions):
        chosen = rng.choice(len(cats), size=infra.n_merchants_per_region, p=cat_p)
        for k, c in enumerate(chosen):
            mid = r * 10000 + k
            merchants[mid] = (cats[c], r)
            merchants_by_region[r].append(mid)

    ips: dict[int, tuple[IpContext, int]] = {}
    public_wifi_by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    corporate_by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    cgnat_by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    for r in range(n_regions):
        for k in range(infra.n_public_wifi_per_region):
            ip = ids.pack(ids.IP_PUBLIC_WIFI, r * 1000 + k)
            ips[ip] = (IpContext.PUBLIC_WIFI, r)
            public_wifi_by_region[r].append(ip)
        for k in range(infra.n_corporate_nat_per_region):
            ip = ids.pack(ids.IP_CORPORATE, r * 1000 + k)
            ips[ip] = (IpContext.CORPORATE_NAT, r)
            corporate_by_region[r].append(ip)
        for k in range(infra.n_cgnat_per_region):
            ip = ids.pack(ids.IP_CGNAT, r * 1000 + k)
            ips[ip] = (IpContext.MOBILE_CGNAT, r)
            cgnat_by_region[r].append(ip)

    # ---- external accounts ----
    externals: dict[ExternalRole, list[int]] = {}
    external_accounts: list[ExternalAccount] = []
    counts = {
        ExternalRole.EMPLOYER: cfg.external.n_employers,
        ExternalRole.BILLER: cfg.external.n_billers,
        ExternalRole.LANDLORD: cfg.external.n_landlords,
        ExternalRole.EXTERNAL_WORLD: cfg.external.n_external_world,
        ExternalRole.CARD_NETWORK: 1,
        ExternalRole.CASH: 1,
    }
    for role_idx, (role, cnt) in enumerate(counts.items()):
        externals[role] = []
        for k in range(cnt):
            acc = ids.pack(ids.ACCOUNT_EXTERNAL, role_idx * 100000 + k)
            externals[role].append(acc)
            external_accounts.append(ExternalAccount(acc, role))

    # ---- persons and households ----
    archetypes = _choose_archetypes(cfg, stream(seed, "population", "persons"))
    rng = stream(seed, "population", "households")
    households: dict[int, dict] = {}
    person_household = np.empty(n, dtype=np.int64)
    person_region = np.empty(n, dtype=np.int64)
    pid = 0
    hh = 0
    while pid < n:
        size = min(sample_int(infra.household_size, rng, minimum=1), n - pid)
        region = int(rng.integers(n_regions))
        members = list(range(pid, pid + size))
        hh_ip = ids.pack(ids.IP_HOUSEHOLD, hh)
        ips[hh_ip] = (IpContext.HOUSEHOLD if size > 1 else IpContext.RESIDENTIAL_SINGLE, region)
        shared = None
        if size > 1 and rng.random() < infra.p_household_shared_device:
            shared = ids.pack(ids.DEVICE_HOUSEHOLD, hh)
        households[hh] = {"region": region, "ip": hh_ip, "shared_device": shared, "members": members}
        person_household[pid : pid + size] = hh
        person_region[pid : pid + size] = region
        pid += size
        hh += 1

    rng = stream(seed, "population", "accounts")
    persons: list[Person] = []
    for i in range(n):
        a = cfg.population.archetypes[archetypes[i]]
        if rng.random() < infra.p_new_account_in_sim:
            opened = int(rng.uniform(0, max(1, tb.days - 14)) * SECONDS_PER_DAY)
        else:
            opened = -int(float(infra.account_age_days.sample(rng)) * SECONDS_PER_DAY)
        overdraft = max(0, int(round(float(a.overdraft_limit.sample(rng)))))
        balance = max(0, int(round(float(a.opening_balance.sample(rng))))) if opened <= 0 else 0
        persons.append(
            Person(
                pid=i,
                archetype=archetypes[i],
                region=int(person_region[i]),
                household=int(person_household[i]),
                account_id=ids.pack(ids.ACCOUNT_INTERNAL, i),
                opened_at=opened,
                overdraft_limit=overdraft,
                initial_balance=balance,
            )
        )

    # ---- devices ----
    rng = stream(seed, "population", "devices")
    devices: dict[int, tuple[DeviceKind, int]] = {}
    device_controllers: dict[int, list[int]] = {}
    for p in persons:
        k_dev = sample_int(infra.devices_per_person, rng, minimum=1)
        first_seen = min(p.opened_at, 0)
        for k in range(k_dev):
            kind = (
                DeviceKind.MOBILE if k == 0 or rng.random() >= infra.p_desktop_device else DeviceKind.DESKTOP
            )
            did = ids.pack(ids.DEVICE_PERSONAL, (p.pid << 4) | k)
            devices[did] = (kind, first_seen)
            device_controllers[did] = [p.pid]
            p.devices.append((did, kind))
    for info in households.values():
        if info["shared_device"] is not None:
            members = info["members"]
            first_seen = min(min(persons[m].opened_at for m in members), 0)
            devices[info["shared_device"]] = (DeviceKind.TABLET, first_seen)
            device_controllers[info["shared_device"]] = list(members)
            for m in members:
                persons[m].devices.append((info["shared_device"], DeviceKind.TABLET))

    # ---- workplace IPs ----
    rng = stream(seed, "population", "ips")
    for p in persons:
        if cfg.population.archetypes[p.archetype].has_workplace_ip:
            pool = corporate_by_region[p.region]
            p.workplace_ip = pool[int(rng.integers(len(pool)))]

    # ---- businesses, employment, landlords ----
    business_pids = [p.pid for p in persons if cfg.population.archetypes[p.archetype].business is not None]
    employers = [q for q in business_pids if _business(cfg, persons[q].archetype).is_employer]
    employer_w = np.asarray(
        [_business(cfg, persons[q].archetype).employer_weight for q in employers],
        dtype=np.float64,
    )
    landlords = [q for q in business_pids if _business(cfg, persons[q].archetype).is_landlord]

    rng = stream(seed, "population", "employment")
    for p in persons:
        inc = cfg.population.archetypes[p.archetype].income
        if inc is None:
            continue
        if employers and rng.random() < inc.p_internal_employer:
            q = employers[int(rng.choice(len(employers), p=employer_w / employer_w.sum()))]
            p.employer_account = persons[q].account_id
            p.employer_is_internal = True
        else:
            pool = externals[ExternalRole.EMPLOYER]
            p.employer_account = pool[int(rng.integers(len(pool)))]

    rng = stream(seed, "population", "landlords")
    for p in persons:
        rent = cfg.population.archetypes[p.archetype].rent
        if rent is None or rng.random() >= rent.p_has_rent:
            continue
        if landlords and rng.random() < rent.p_internal_landlord:
            q = landlords[int(rng.integers(len(landlords)))]
            if q != p.pid:
                p.landlord_account = persons[q].account_id
                p.landlord_is_internal = True
                continue
        pool = externals[ExternalRole.LANDLORD]
        p.landlord_account = pool[int(rng.integers(len(pool)))]

    # ---- contacts (P2P graph) and favorite merchants ----
    rng = stream(seed, "population", "contacts")
    by_region: dict[int, list[int]] = {r: [] for r in range(n_regions)}
    for p in persons:
        by_region[p.region].append(p.pid)
    for p in persons:
        a = cfg.population.archetypes[p.archetype]
        hh_members = [m for m in households[p.household]["members"] if m != p.pid]
        contacts = list(hh_members)
        if a.p2p is not None:
            k = sample_int(a.p2p.n_contacts, rng, minimum=1)
            for _ in range(k):
                if rng.random() < a.p2p.p_same_region_contact:
                    pool = by_region[p.region]
                    c = pool[int(rng.integers(len(pool)))]
                else:
                    c = int(rng.integers(n))
                if c != p.pid and c not in contacts:
                    contacts.append(c)
        p.contacts = contacts
        if a.card is not None:
            pool = merchants_by_region[p.region]
            k = min(a.card.n_favorite_merchants, len(pool))
            p.favorite_merchants = [int(x) for x in rng.choice(pool, size=k, replace=False)]

    account_to_pid = {p.account_id: p.pid for p in persons}
    return Population(
        timebase=tb,
        persons=persons,
        households=households,
        externals=externals,
        external_accounts=external_accounts,
        devices=devices,
        ips=ips,
        atms=atms,
        merchants=merchants,
        atms_by_region=atms_by_region,
        merchants_by_region=merchants_by_region,
        public_wifi_by_region=public_wifi_by_region,
        corporate_by_region=corporate_by_region,
        cgnat_by_region=cgnat_by_region,
        business_pids=business_pids,
        account_to_pid=account_to_pid,
        device_controllers=device_controllers,
    )
