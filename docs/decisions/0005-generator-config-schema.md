# 0005: Generator configuration schema

- Status: PROPOSED (awaiting review, Phase 0)
- Scope: `configs/generator/*.yaml` and their Pydantic schema, Milestone 1

## Problem

The generator has many parameters (population sizes, archetype mixes, amount and timing
distributions, device/IP sharing, confounders, scenario families, held-out families, label
delays). They must be validated, versioned, hashed into dataset metadata, and composed into
profiles (DEV, RESEARCH, later LARGE) without duplication. Invalid config must fail loudly.

## Options

- **A. Hydra + OmegaConf structured configs.** Powerful composition and sweeps. Costs: changes the
  working directory by default, an additional config language, interpolation that can hide values,
  heavier dependency; its strength (multirun sweeps) is not needed in Milestone 1.
- **B. Pydantic v2 models + YAML files + a small explicit overlay merge.** Validation, clear error
  messages, JSON-schema export, trivially hashable (`model_dump_json` with sorted keys).
- **C. Plain dataclasses + YAML.** Weak validation.

## Recommendation

**B** now; revisit Hydra (or not) at Milestone 3 when experiment sweeps appear.

Rules:

1. All models `model_config = ConfigDict(extra="forbid", frozen=True)`. Unknown keys raise.
2. Profiles are overlays: `base.yaml` plus `dev.yaml` or `research.yaml`; merge is a documented
   deep-merge of mappings where lists are replaced, never concatenated. The resolved config is
   written to metadata and hashed.
3. Distributions are typed, discriminated unions (`LogNormal(mu, sigma)`, `Gamma(k, theta)`,
   `Empirical(values, probs)` ...), never free-form strings.
4. Probabilities validated to be in [0, 1] and mixtures to sum to 1 within 1e-9.
5. Sizes are explicit per profile; no hidden defaults that differ between profiles.

Sketch (illustrative, not final code):

```yaml
simulation:
  seed: 42
  days: 90
  calendar_anchor: 2025-01-06   # a Monday; only for seasonality
population:
  n_persons: ...
  archetype_mix: {salary_worker: 0.35, student: 0.12, small_business: 0.08, ...}
infrastructure:
  shared_ip_contexts: {household: ..., public_wifi: ..., corporate_nat: ..., mobile_cgnat: ...}
confounders:
  legit_high_volume_business: {share: ..., fan_in_degree: Gamma(...), ...}
  temporary_spending_surge: {...}
scenarios:
  - family: fan_in
    enabled: true
    n_instances: ...
    split_role: train_eligible        # or held_out
    shape: {n_sources: ..., duration: ..., hold_time: ...}
    phases: {transition: ..., durations: ...}
labels:
  horizons_hours: [24, 168]
  knowledge_delay: {kind: LogNormal, ...}
```

### Safety rule embedded in the schema

Scenario parameters describe **pattern shape only** (degree, hop count, durations, amount
distribution *shape*, timing). The schema has no field whose meaning is "stay below a detection
threshold", no field that references a detector, and the generator never reads model scores.
There is no automated search over scenario parameters against any detector (see
`docs/SECURITY_AND_ETHICS.md`). The structuring-like family is expressed as "many transfers of
similar size clustered in a configured amount band", which is the *detection-side* description of
the pattern.

## Trade-offs

- No sweeps or command-line override syntax out of the box; a minimal `--set key=value` override
  can be added later if needed, validated through the same schema.
- Discriminated unions for many families produce verbose schemas; accepted for strictness.
