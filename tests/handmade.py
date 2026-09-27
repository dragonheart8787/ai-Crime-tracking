"""A tiny hand-built dataset for exact leakage and label-knowledge tests (decision 0004 required test).

Accounts: A and C are scenario members; B is A's normal counterparty; D is C's counterparty; X is external.
Network 1 (A): inflow B->A on day 10, terminal A->X on day 20, known_at day 70.
Network 2 (C, control): inflow D->C on day 5, terminal C->X on day 7, known_at day 30.
"""

from __future__ import annotations

import pyarrow as pa

from fcip.common.schemas import conform

DAY = 86400
A, B, C, D, X = 10, 11, 12, 13, 99
SIM_DAYS = 90


def tables() -> dict[str, pa.Table]:
    tx = [  # event_id, ts, type, channel, src, dst, amount, status, balances
        (1, 10 * DAY, "TRANSFER", "SCHEDULED", B, A, 500, 1000, 500, 0, 500),
        (2, 20 * DAY, "OUTBOUND_PAYMENT", "SCHEDULED", A, X, 400, 500, 100, 0, 400),
        (3, 5 * DAY, "TRANSFER", "SCHEDULED", D, C, 300, 1000, 700, 0, 300),
        (4, 7 * DAY, "OUTBOUND_PAYMENT", "SCHEDULED", C, X, 250, 300, 50, 400, 650),
        (5, 64 * DAY, "TRANSFER", "SCHEDULED", A, B, 10, 100, 90, 500, 510),
    ]
    t = pa.table(
        {
            "event_id": [r[0] for r in tx],
            "ts": [r[1] for r in tx],
            "txn_type": [r[2] for r in tx],
            "channel": [r[3] for r in tx],
            "src_account_id": [r[4] for r in tx],
            "dst_account_id": [r[5] for r in tx],
            "merchant_id": [None] * 5,
            "atm_id": [None] * 5,
            "amount_minor": [r[6] for r in tx],
            "region": [0] * 5,
            "device_id": [None] * 5,
            "ip_id": [None] * 5,
            "login_event_id": [None] * 5,
            "status": ["SETTLED"] * 5,
            "src_balance_before_minor": [r[7] for r in tx],
            "src_balance_after_minor": [r[8] for r in tx],
            "dst_balance_before_minor": [r[9] for r in tx],
            "dst_balance_after_minor": [r[10] for r in tx],
        }
    )
    accts = [A, B, C, D, X]
    acc = pa.table(
        {
            "account_id": accts,
            "account_kind": ["internal"] * 4 + ["external"],
            "external_role": [None] * 4 + ["external_world"],
            "owner_person_id": [1, 2, 3, 4, None],
            "opened_at": [-DAY] * 5,
            "closed_at": [None] * 5,
            "overdraft_limit_minor": [0] * 4 + [None],
            "initial_balance_minor": [0, 1000, 0, 1000, 0],
            "region": [0] * 4 + [None],
        }
    )
    lab = pa.table(
        {
            "entity_type": ["account"] * 2,
            "entity_id": [A, C],
            "valid_from": [10 * DAY, 5 * DAY],
            "valid_to": [21 * DAY, 8 * DAY],
            "risk_label": [1, 1],
            "role_label": ["FUND_RECEIVER", "FUND_RECEIVER"],
            "scenario_id": ["fan_in:0", "fan_in:1"],
            "network_id": [1, 2],
            "family": ["fan_in", "fan_in"],
            "phase": ["INFLOW", "INFLOW"],
            "is_ood_family": [False, False],
            "known_at": [70 * DAY, 30 * DAY],
        }
    )
    ev = pa.table(
        {
            "event_table": ["transactions"] * 4,
            "event_id": [1, 2, 3, 4],
            "scenario_id": ["fan_in:0"] * 2 + ["fan_in:1"] * 2,
            "network_id": [1, 1, 2, 2],
            "family": ["fan_in"] * 4,
            "phase": ["INFLOW", "EXIT", "INFLOW", "EXIT"],
            "event_type": ["RECEIVE_FUNDS", "TRANSFER_FUNDS"] * 2,
            "is_high_risk": [False, True, False, True],
            "is_terminal": [False, True, False, True],
            "is_ood_family": [False] * 4,
            "known_at": [70 * DAY] * 2 + [30 * DAY] * 2,
        }
    )
    nets = pa.table(
        {
            "network_id": [1, 2],
            "scenario_id": ["fan_in:0", "fan_in:1"],
            "family": ["fan_in"] * 2,
            "instance_idx": [0, 1],
            "is_ood_family": [False, False],
            "start_ts": [10 * DAY, 5 * DAY],
            "end_ts": [21 * DAY, 8 * DAY],
            "terminal_ts": [20 * DAY, 7 * DAY],
            "censored": [False, False],
            "n_members": [1, 1],
            "n_created_accounts": [0, 0],
            "n_rounds": [1, 1],
            "detected": [True, True],
            "known_at": [70 * DAY, 30 * DAY],
            "amount_scale_minor": [500, 300],
            "fan_degree": [1, 1],
            "hops": [1, 1],
            "holding_seconds": [0, 0],
            "retention_permille": [0, 0],
            "forward_permille": [800, 830],
        }
    )
    mem = pa.table(
        {
            "network_id": [1, 2],
            "entity_type": ["account"] * 2,
            "entity_id": [A, C],
            "first_role": ["FUND_RECEIVER"] * 2,
            "joined_ts": [10 * DAY, 5 * DAY],
            "is_created": [False, False],
        }
    )
    empty = {
        "logins": pa.table(
            {
                "event_id": pa.array([], pa.int64()),
                "ts": pa.array([], pa.int64()),
                "account_id": pa.array([], pa.int64()),
                "device_id": pa.array([], pa.int64()),
                "ip_id": pa.array([], pa.int64()),
                "channel": pa.array([], pa.string()),
                "outcome": pa.array([], pa.string()),
            }
        ),
        "relations": pa.table(
            {
                "relation_type": ["OWNS_ACCOUNT"] * 4,
                "src_id": [1, 2, 3, 4],
                "dst_id": [A, B, C, D],
                "valid_from": [-DAY] * 4,
                "valid_to": [SIM_DAYS * DAY] * 4,
            }
        ),
        "devices": pa.table(
            {
                "device_id": pa.array([], pa.int64()),
                "device_kind": pa.array([], pa.string()),
                "first_seen_at": pa.array([], pa.int64()),
            }
        ),
        "ips": pa.table(
            {
                "ip_id": pa.array([], pa.int64()),
                "ip_context": pa.array([], pa.string()),
                "region": pa.array([], pa.int16()),
            }
        ),
        "atms": pa.table({"atm_id": pa.array([], pa.int64()), "region": pa.array([], pa.int16())}),
        "merchants": pa.table(
            {
                "merchant_id": pa.array([], pa.int64()),
                "merchant_category": pa.array([], pa.string()),
                "region": pa.array([], pa.int16()),
            }
        ),
        "persons": pa.table(
            {
                "person_id": [1, 2, 3, 4],
                "region": [0] * 4,
                "household_id": [1, 2, 3, 4],
                "created_at": [-DAY] * 4,
            }
        ),
    }
    raw = {
        "transactions": t,
        "accounts": acc,
        "labels": lab,
        "event_labels": ev,
        "ground_truth_networks": nets,
        "network_members": mem,
        **empty,
    }
    return {k: conform(k, v) for k, v in raw.items()}
