# Security and Ethics

Status: initial version (Phase 0). Maintained for the lifetime of the project (`CLAUDE.md` 2.1).

## Purpose

This project is defensive research: simulating, detecting, explaining and evaluating anomalous
financial patterns in **synthetic** data, as decision support for human analysts.

## Language

- Outputs describe *patterns and entities in synthetic data*, never people. Allowed vocabulary:
  suspicious behavior, anomalous financial pattern, high-risk entity / transaction / network,
  predicted high-risk financial event, model confidence, uncertainty.
- Role labels (e.g. RELAY, AGGREGATOR) are names for observed behavior inside the synthetic graph.
  They are not legal conclusions and must not be presented as such in any report, alert or UI.
- No output states or implies that a person is a criminal or guilty.

## Decision support only

AI produces risk intelligence; a human analyst decides. The system never automates accusation,
account closure, reporting or any other consequential action. Any future analyst-feedback loop must
not retrain models online without explicit safeguards and review.

## No evasion capability

The project must not contain functionality that helps anyone evade monitoring. Concretely:

1. Scenario parameters describe pattern *shape* (degree, hop count, durations, amount-distribution
   shape, timing). No parameter means "stay below a detection threshold" or references a detector.
2. The generator never reads model scores, alerts or detector outputs. There is no loop, script or
   notebook that searches scenario parameters to reduce detection (no adversarial tuning against our
   own or any other detector), no "safe amount" search, and no route optimization.
3. Structuring-like patterns are represented by their detection-side description (clusters of
   similar-sized transfers in an amount band), not by rules about real reporting thresholds.
4. Reports and docs describe patterns at the level needed for detection research, not as instructions.

Any proposal that would require (2), for example adversarial robustness studies, needs an explicit
written decision record and approval first, and by default is out of scope.

## Weak signals

A single weak signal (for example a shared IP address or shared device) must never by itself imply
suspicion. The generator deliberately includes legitimate sharing (households, public Wi-Fi, corporate
and carrier NAT, family devices), and evaluation reports the false-positive share attributable to such
confounders.

## Data

- Primary data is synthetic and contains no real personal data.
- Before any public dataset is used, a data card records its license, provenance, whether it is real
  or synthetic, schema, labels and limitations. Real personal financial data is out of scope.
- Generated datasets are not committed to git (`.gitignore`), and are reproducible from config + seed.

## Review checklist (applied at every milestone review packet)

- [ ] No new parameter or code path that optimizes against detection.
- [ ] Generator has no dependency on any model or scoring module.
- [ ] Wording of new outputs, reports and UI text follows the language rules above.
- [ ] No single weak signal is used as a standalone alert rule.
- [ ] Any new external data source has a data card.
