"""Column-oriented event log used while generating (transactions and logins), before settlement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fcip.common.taxonomy import Channel, EventType, LoginOutcome, Phase, TxnType

NO_NETWORK = -1


@dataclass
class EventLog:
    # transactions
    t_event_id: list[int] = field(default_factory=list)
    t_ts: list[int] = field(default_factory=list)
    t_type: list[str] = field(default_factory=list)
    t_channel: list[str] = field(default_factory=list)
    t_src: list[int] = field(default_factory=list)
    t_dst: list[int] = field(default_factory=list)
    t_merchant: list[int | None] = field(default_factory=list)
    t_atm: list[int | None] = field(default_factory=list)
    t_amount: list[int] = field(default_factory=list)
    t_region: list[int] = field(default_factory=list)
    t_device: list[int | None] = field(default_factory=list)
    t_ip: list[int | None] = field(default_factory=list)
    t_login: list[int | None] = field(default_factory=list)
    t_depends_on: list[int] = field(default_factory=list)
    t_network: list[int] = field(default_factory=list)
    t_phase: list[str | None] = field(default_factory=list)
    t_event_type: list[str | None] = field(default_factory=list)
    # logins
    l_event_id: list[int] = field(default_factory=list)
    l_ts: list[int] = field(default_factory=list)
    l_account: list[int] = field(default_factory=list)
    l_device: list[int] = field(default_factory=list)
    l_ip: list[int] = field(default_factory=list)
    l_channel: list[str] = field(default_factory=list)
    l_outcome: list[str] = field(default_factory=list)
    l_network: list[int] = field(default_factory=list)
    l_phase: list[str | None] = field(default_factory=list)
    l_event_type: list[str | None] = field(default_factory=list)

    def add_txn(
        self,
        *,
        event_id: int,
        ts: int,
        txn_type: TxnType,
        channel: Channel,
        src: int,
        dst: int,
        amount: int,
        region: int,
        merchant: int | None = None,
        atm: int | None = None,
        device: int | None = None,
        ip: int | None = None,
        login: int | None = None,
        depends_on: int = -1,
        network: int = NO_NETWORK,
        phase: Phase | None = None,
        event_type: EventType | None = None,
    ) -> int:
        if amount <= 0:
            raise ValueError(f"transaction amounts must be positive, got {amount}")
        if src == dst:
            raise ValueError("transaction source and destination must differ")
        self.t_event_id.append(event_id)
        self.t_ts.append(int(ts))
        self.t_type.append(txn_type.value)
        self.t_channel.append(channel.value)
        self.t_src.append(src)
        self.t_dst.append(dst)
        self.t_merchant.append(merchant)
        self.t_atm.append(atm)
        self.t_amount.append(int(amount))
        self.t_region.append(int(region))
        self.t_device.append(device)
        self.t_ip.append(ip)
        self.t_login.append(login)
        self.t_depends_on.append(depends_on)
        self.t_network.append(network)
        self.t_phase.append(None if phase is None else phase.value)
        self.t_event_type.append(None if event_type is None else event_type.value)
        return event_id

    def add_login(
        self,
        *,
        event_id: int,
        ts: int,
        account: int,
        device: int,
        ip: int,
        channel: Channel,
        outcome: LoginOutcome,
        network: int = NO_NETWORK,
        phase: Phase | None = None,
        event_type: EventType | None = None,
    ) -> int:
        self.l_event_id.append(event_id)
        self.l_ts.append(int(ts))
        self.l_account.append(account)
        self.l_device.append(device)
        self.l_ip.append(ip)
        self.l_channel.append(channel.value)
        self.l_outcome.append(outcome.value)
        self.l_network.append(network)
        self.l_phase.append(None if phase is None else phase.value)
        self.l_event_type.append(None if event_type is None else event_type.value)
        return event_id

    def extend(self, other: EventLog) -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).extend(getattr(other, name))

    @property
    def n_txn(self) -> int:
        return len(self.t_event_id)

    @property
    def n_login(self) -> int:
        return len(self.l_event_id)

    def filter(self, keep_txn: np.ndarray, keep_login: np.ndarray) -> EventLog:
        out = EventLog()
        tk = np.flatnonzero(keep_txn)
        lk = np.flatnonzero(keep_login)
        for name in self.__dataclass_fields__:
            src = getattr(self, name)
            idx = tk if name.startswith("t_") else lk
            setattr(out, name, [src[i] for i in idx])
        return out
