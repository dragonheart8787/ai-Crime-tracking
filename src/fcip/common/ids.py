"""Namespaced int64 identifiers (decision 0002). No global counters: every ID is a pure function of
the entity or stream that owns it, so adding a scenario never renumbers unrelated entities or events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Entity-kind prefixes live in bits 48..55 of the ID.
_KIND_SHIFT = 48
_PAYLOAD_MASK = (1 << _KIND_SHIFT) - 1

ACCOUNT_INTERNAL = 1
ACCOUNT_EXTERNAL = 2
ACCOUNT_SCENARIO = 3

PERSON_NORMAL = 0
PERSON_SCENARIO = 3

DEVICE_PERSONAL = 1
DEVICE_HOUSEHOLD = 2
DEVICE_SCENARIO = 3

IP_HOUSEHOLD = 1
IP_PUBLIC_WIFI = 2
IP_CORPORATE = 3
IP_CGNAT = 4
IP_RESIDENTIAL = 5
IP_SCENARIO = 6


def pack(kind: int, payload: int) -> int:
    if not 0 <= kind < 128:
        raise ValueError(f"kind out of range: {kind}")
    if not 0 <= payload <= _PAYLOAD_MASK:
        raise ValueError(f"payload out of range: {payload}")
    return (kind << _KIND_SHIFT) | payload


def kind_of(entity_id: int) -> int:
    return entity_id >> _KIND_SHIFT


def scenario_payload(family_code: int, instance_idx: int, local_idx: int) -> int:
    """Payload for scenario-namespace entities: family (8 bits), instance (16 bits), local (16 bits)."""
    if not (0 < family_code < 256 and 0 <= instance_idx < 1 << 16 and 0 <= local_idx < 1 << 16):
        raise ValueError(f"scenario payload out of range: {family_code}, {instance_idx}, {local_idx}")
    return (family_code << 32) | (instance_idx << 16) | local_idx


def network_id(family_code: int, instance_idx: int) -> int:
    return (family_code << 16) | instance_idx


# Event-ID namespaces (bits 59..62).
EVENT_NORMAL_TXN = 1
EVENT_NORMAL_LOGIN = 2
EVENT_SCENARIO_TXN = 3
EVENT_SCENARIO_LOGIN = 4

_EVENT_NS_SHIFT = 59
_EVENT_ENTITY_SHIFT = 23
_EVENT_COUNTER_MAX = (1 << _EVENT_ENTITY_SHIFT) - 1
_EVENT_ENTITY_MAX = (1 << (_EVENT_NS_SHIFT - _EVENT_ENTITY_SHIFT)) - 1


@dataclass
class EventIdAllocator:
    """Allocates event IDs ``(namespace, owner entity, local counter)`` for one generating stream."""

    namespace: int
    entity: int
    _next: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.namespace < 16:
            raise ValueError(f"event namespace out of range: {self.namespace}")
        if not 0 <= self.entity <= _EVENT_ENTITY_MAX:
            raise ValueError(f"event owner entity out of range: {self.entity}")

    def next(self) -> int:
        if self._next > _EVENT_COUNTER_MAX:
            raise OverflowError(
                f"event counter exhausted for namespace {self.namespace} entity {self.entity}"
            )
        eid = (self.namespace << _EVENT_NS_SHIFT) | (self.entity << _EVENT_ENTITY_SHIFT) | self._next
        self._next += 1
        return eid


def event_namespace(event_id: int) -> int:
    return event_id >> _EVENT_NS_SHIFT
