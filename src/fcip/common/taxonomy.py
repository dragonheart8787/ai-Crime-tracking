"""Closed vocabularies used by the generator and the output schemas.

Roles and event types follow CLAUDE.md section 5. They describe observed behavior inside the synthetic
graph, never legal conclusions.
"""

from enum import StrEnum


class Archetype(StrEnum):
    SALARY_WORKER = "salary_worker"
    STUDENT = "student"
    SMALL_BUSINESS = "small_business"
    HF_MERCHANT = "hf_merchant"
    TRAVELER = "traveler"
    FAMILY = "family"
    SPENDING_SURGE = "spending_surge"
    HIGH_VOLUME_BUSINESS = "high_volume_business"


class Family(StrEnum):
    """Abstract suspicious-pattern families (detection research patterns only)."""

    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"
    PASS_THROUGH = "pass_through"
    BURST = "burst"
    DORMANT_ACTIVATION = "dormant_activation"
    MULTI_HOP = "multi_hop"
    CYCLE = "cycle"
    STRUCTURING_LIKE = "structuring_like"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"
    ACCOUNT_TO_CASH = "account_to_cash"


# Stable small integer codes, used inside packed IDs. Never reorder; append only.
FAMILY_CODE: dict[Family, int] = {f: i + 1 for i, f in enumerate(Family)}


class Phase(StrEnum):
    SETUP = "SETUP"
    INFLOW = "INFLOW"
    HOLD = "HOLD"
    MOVEMENT = "MOVEMENT"
    EXIT = "EXIT"
    INACTIVE = "INACTIVE"


class Role(StrEnum):
    NORMAL = "NORMAL"
    VICTIM_LIKE = "VICTIM_LIKE"
    FUND_RECEIVER = "FUND_RECEIVER"
    RELAY = "RELAY"
    AGGREGATOR = "AGGREGATOR"
    DISTRIBUTOR = "DISTRIBUTOR"
    CASH_OUT_RISK = "CASH_OUT_RISK"
    SUSPICIOUS_UNKNOWN = "SUSPICIOUS_UNKNOWN"


class EventType(StrEnum):
    RECEIVE_FUNDS = "RECEIVE_FUNDS"
    TRANSFER_FUNDS = "TRANSFER_FUNDS"
    MULTI_TRANSFER = "MULTI_TRANSFER"
    ATM_WITHDRAWAL = "ATM_WITHDRAWAL"
    CARD_PAYMENT = "CARD_PAYMENT"
    NEW_BENEFICIARY = "NEW_BENEFICIARY"
    NEW_DEVICE_LOGIN = "NEW_DEVICE_LOGIN"
    NEW_IP_LOGIN = "NEW_IP_LOGIN"
    BALANCE_HOLD = "BALANCE_HOLD"
    MERCHANT_PAYMENT = "MERCHANT_PAYMENT"
    NO_ACTIVITY = "NO_ACTIVITY"
    OTHER = "OTHER"


class Channel(StrEnum):
    APP = "APP"
    WEB = "WEB"
    ATM = "ATM"
    POS = "POS"
    SCHEDULED = "SCHEDULED"
    INBOUND_EXTERNAL = "INBOUND_EXTERNAL"


DIGITAL_CHANNELS: frozenset[Channel] = frozenset({Channel.APP, Channel.WEB})


class TxnType(StrEnum):
    TRANSFER = "TRANSFER"
    CARD_PAYMENT = "CARD_PAYMENT"
    ATM_WITHDRAWAL = "ATM_WITHDRAWAL"
    INBOUND_CREDIT = "INBOUND_CREDIT"
    OUTBOUND_PAYMENT = "OUTBOUND_PAYMENT"


# Which channels are valid for which transaction type (checked as an invariant).
ALLOWED_CHANNELS: dict[TxnType, frozenset[Channel]] = {
    TxnType.TRANSFER: frozenset({Channel.APP, Channel.WEB, Channel.SCHEDULED}),
    TxnType.CARD_PAYMENT: frozenset({Channel.POS}),
    TxnType.ATM_WITHDRAWAL: frozenset({Channel.ATM}),
    TxnType.INBOUND_CREDIT: frozenset({Channel.INBOUND_EXTERNAL}),
    TxnType.OUTBOUND_PAYMENT: frozenset({Channel.SCHEDULED, Channel.APP, Channel.WEB}),
}


class Status(StrEnum):
    SETTLED = "SETTLED"
    DECLINED = "DECLINED"


class LoginOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class IpContext(StrEnum):
    HOUSEHOLD = "household"
    PUBLIC_WIFI = "public_wifi"
    CORPORATE_NAT = "corporate_nat"
    MOBILE_CGNAT = "mobile_cgnat"
    RESIDENTIAL_SINGLE = "residential_single"


class DeviceKind(StrEnum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class ExternalRole(StrEnum):
    EMPLOYER = "employer"
    BILLER = "biller"
    LANDLORD = "landlord"
    EXTERNAL_WORLD = "external_world"
    CARD_NETWORK = "card_network"
    CASH = "cash"


class RelationType(StrEnum):
    OWNS_ACCOUNT = "OWNS_ACCOUNT"
    CONTROLS = "CONTROLS"
    USES_IP = "USES_IP"
