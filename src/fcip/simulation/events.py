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


# ---------------------------------------------------------------------------------------------------------
# Compact storage. Python lists of ints cost about 40 bytes per value; at RESEARCH scale (about 3e7 events)
# that does not fit in memory, so logs are converted to numpy columns in batches. Enumerated string columns
# become int8 codes; nullable integer columns use -1 as the null sentinel (all IDs are non-negative).
# ---------------------------------------------------------------------------------------------------------
TXN_TYPES: tuple[str, ...] = tuple(t.value for t in TxnType)
CHANNELS: tuple[str, ...] = tuple(c.value for c in Channel)
PHASES: tuple[str, ...] = tuple(p.value for p in Phase)
EVENT_TYPES: tuple[str, ...] = tuple(e.value for e in EventType)
OUTCOMES: tuple[str, ...] = tuple(o.value for o in LoginOutcome)
NULL = -1

_T_INT = (
    "t_event_id",
    "t_ts",
    "t_src",
    "t_dst",
    "t_merchant",
    "t_atm",
    "t_amount",
    "t_region",
    "t_device",
    "t_ip",
    "t_login",
    "t_depends_on",
    "t_network",
)
_T_CODE = {"t_type": TXN_TYPES, "t_channel": CHANNELS, "t_phase": PHASES, "t_event_type": EVENT_TYPES}
_L_INT = ("l_event_id", "l_ts", "l_account", "l_device", "l_ip", "l_network")
_L_CODE = {"l_channel": CHANNELS, "l_outcome": OUTCOMES, "l_phase": PHASES, "l_event_type": EVENT_TYPES}


def _codes(values: list[str | None], vocab: tuple[str, ...]) -> np.ndarray:
    index = {v: i for i, v in enumerate(vocab)}
    return np.fromiter((NULL if v is None else index[v] for v in values), dtype=np.int8, count=len(values))


def _ints(values: list[int | None]) -> np.ndarray:
    return np.fromiter((NULL if v is None else v for v in values), dtype=np.int64, count=len(values))


@dataclass
class EventArrays:
    """Numpy-column form of :class:`EventLog` (same field names)."""

    cols: dict[str, np.ndarray]

    @classmethod
    def from_log(cls, log: EventLog) -> EventArrays:
        cols: dict[str, np.ndarray] = {}
        for name in _T_INT + _L_INT:
            cols[name] = _ints(getattr(log, name))
        for name, vocab in {**_T_CODE, **_L_CODE}.items():
            cols[name] = _codes(getattr(log, name), vocab)
        return cls(cols)

    @classmethod
    def concat(cls, parts: list[EventArrays]) -> EventArrays:
        if not parts:
            return cls.from_log(EventLog())
        return cls({k: np.concatenate([p.cols[k] for p in parts]) for k in parts[0].cols})

    @property
    def n_txn(self) -> int:
        return len(self.cols["t_event_id"])

    @property
    def n_login(self) -> int:
        return len(self.cols["l_event_id"])

    def filter(self, keep_txn: np.ndarray, keep_login: np.ndarray) -> EventArrays:
        return EventArrays(
            {k: (v[keep_txn] if k.startswith("t_") else v[keep_login]) for k, v in self.cols.items()}
        )


def decode(codes: np.ndarray, vocab: tuple[str, ...]) -> list[str | None]:
    return [None if c == NULL else vocab[c] for c in codes.tolist()]
