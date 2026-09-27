# 0010: Milestone 1 generator implementation choices (checkpoint 1)

- Status: PROPOSED (for review with the checkpoint 1 report)
- Scope: `src/fcip/` as built at Milestone 1 checkpoint 1; records where the implementation differs from the
  Phase 0 plan and why. Each item follows the decision protocol in short form.

## 1. One generic behavior engine instead of one module per archetype

- Problem: eight archetypes share most behaviors (income, rent, bills, card, ATM, P2P, logins) and differ in
  parameters.
- Options: (a) a module per archetype (planned in `PHASE0_ASSESSMENT.md` section 8); (b) one engine driven by
  per-archetype config blocks.
- Decision: (b), `simulation/behavior.py`. Archetypes differ only through config (`population.archetypes`),
  which keeps behavior assumptions visible in YAML rather than spread across modules.
- Trade-off: one larger module (about 400 lines); archetype-specific logic beyond the shared blocks (travel, surge,
  dormancy, business flows) lives in optional config blocks.

## 2. Archetype and origin moved to an oracle table `person_truth`

- Problem: the planned `persons.archetype` column is generator ground truth, not something a bank observes.
  Scenario-created persons would also be marked in it, which would make them trivially identifiable.
- Decision: `persons` has only observable attributes (region, household, creation time). Archetype and origin
  (`normal` / `scenario`) live in `person_truth`, registered as an oracle table like the label tables.

## 3. Every transaction has a destination account; card and cash flows go to external sink accounts

- Problem: conservation of funds needs every settled event to move money between two ledger accounts.
- Decision: `dst_account_id` is non-null. Card payments credit an external `card_network` account (with the
  merchant in `merchant_id`); ATM withdrawals credit an external `cash` account (with the ATM in `atm_id`).
  External accounts have no overdraft limit.

## 4. Card-acquired business revenue settles as one credit per day

- Problem: first runs produced one inbound credit per external customer payment, which dominated volume and is not
  how acquiring works.
- Decision: external customer payments to a business are summed per day into one `INBOUND_CREDIT` at about 23:00
  from that business's acquirer. Internal customer payments stay individual `APP` transfers (they create the
  high in-degree confounder).

## 5. Label table details

- No `INACTIVE` rows: absence of a label row means not in an active scenario phase. The pre-registered gate target
  ("phase other than INACTIVE") is unaffected.
- Role for an `EXIT` to an external account is `RELAY`; `CASH_OUT_RISK` is used only for ATM cash-out.
- `SETUP` rows use `SUSPICIOUS_UNKNOWN` (members are recruited, but their functional role is not yet exercised).
- Victims (`VICTIM_LIKE`, `risk_label = 0`) are network members and count toward prevalence, as defined in 0005.
- Networks that never exit (the `p_exit` draw fails) have no terminal event and are not censored ("abandoned");
  `known_at` is then anchored on their last settled event (decision 0004 rule for censored networks, applied to both).
- `label_knowledge.regime: oracle` sets `known_at` to the anchor time (zero latency).

## 6. Suppression of members' normal activity

- Retention thinning and dormancy apply to every normal event that involves the member's account (as source,
  destination or login account) within the window, using a keyed hash of `event_id` (no RNG state), and remove the
  session logins of removed transactions.

## 7. Loader conveniences

- `family_defaults` (deep-merged under every family) and top-level keys starting with `_` (YAML anchor holders only)
  are handled by the loader; both are removed before validation, so the validated config is fully explicit and is
  what gets hashed.
- `infrastructure.session_window_seconds` (1,800 s) is the explicit bound used by the login invariant;
  `session_seconds` must lie within it.

## 8. Default OOD families

`cycle` and `dormant_activation` are marked `is_ood_family: true` in `base.yaml`: one topological novelty and one
temporal novelty. This is a default for review, not a research decision.

## 9. Language verdicts for Milestone 1 components (review cadence from 0006)

Measured with `cProfile` on one DEV generation (2,500 persons, 90 days, 327K transactions, 431K logins;
23.2 s profiled, about 16 s unprofiled):

| Component | Measured cost | Verdict |
|---|---|---|
| normal behavior (`generate_person`) | 16.0 s cumulative (69%) | Python; first step if RESEARCH is too slow is vectorizing per-day Poisson loops in numpy, not Rust |
| settlement pass (`settle`) | 1.1 s (about 3.4 µs per event) | Python only; the Phase 0 concern (0006 a') is not borne out at this scale |
| invariant checks | 1.4 s | Python (polars) |
| content hashing | 1.3 s | Python (vectorized) |
| recruitment partition, suppression | < 1 s each | Python |

RESEARCH generation time is NOT YET EVALUATED (not run at checkpoint 1). A linear extrapolation (40 times the persons,
16 s x 40) gives roughly 11 minutes, close to the 15-minute re-evaluation trigger in 0006 and to the Q-R4 runtime
threshold, so it must be measured before the gate run.

## 10. Measured RESEARCH generation (A5, Milestone 1 checkpoint 2)

Measured, not extrapolated: RESEARCH profile, calibration seed 1000, one run in this 4-CPU, 15 GB cloud sandbox,
with the compact numpy event store and chunked settlement introduced at checkpoint 2 (they cut DEV peak memory from
0.77 GB to 0.50 GB with a byte-identical DEV dataset; without them RESEARCH would have needed roughly 31 GB).

| Quantity | Value |
|---|---|
| wall-clock time (generation, invariant checks, hashing, Parquet writing) | **540.6 s (9.0 min)** |
| peak resident memory | 11.82 GB |
| transactions / logins | 12,852,147 / 16,830,023 |
| realized prevalence | 0.01001 (target 0.01) |

Q-R4 rule: 9.0 minutes is below the 15-minute threshold, so the gates run on the **RESEARCH** profile (not the 25K
GATE profile). No Rust port is warranted by this number (decision 0006 trigger: 15 minutes or 24 GB). Peak memory at
11.8 GB is the tighter constraint in this sandbox; the Windows laptop has 64 GB.

Two RESEARCH-only failures were found and fixed before this measurement: (1) `max_instances: 40` for every family
could not meet the 1% quota for small-instance families (pass_through reached 40 instances with 65 of 100 members;
the generator raised as designed) and was replaced by per-family values; (2) the detection-anchor rule allowed a
label row to start after `known_at` (decision 0004, revised).
