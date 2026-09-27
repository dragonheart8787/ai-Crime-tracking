# 0005: Generator configuration schema

- Status: ACCEPTED (Phase 0 review), revised in the Phase 0 revision pass (`suspicious_prevalence`,
  per-instance parameter distributions, OOD family flag, label-knowledge block, login block,
  recruitment `max_instances`)
- Scope: `configs/generator/*.yaml` and their Pydantic schema, Milestone 1

## Problem

The generator has many parameters that must be validated, versioned, hashed into dataset metadata and composed
into profiles (DEV, RESEARCH, later LARGE) without duplication. Invalid config must fail loudly.

## Options

- A. Hydra + OmegaConf structured configs: composition and sweeps, but changes the working directory by default,
  adds an interpolation language that can hide values, and sweeps are not needed in Milestone 1.
- B. **Pydantic v2 models + YAML + a small explicit overlay merge.**
- C. Plain dataclasses + YAML: weak validation.

## Decision

**B** now; revisit Hydra at Milestone 3 when experiment sweeps appear.

Rules:

1. All models `model_config = ConfigDict(extra="forbid", frozen=True)`. Unknown keys raise.
2. Profiles are overlays: `base.yaml` plus `dev.yaml` or `research.yaml`; documented deep-merge of mappings,
   lists replaced, never concatenated. The resolved config is written to metadata and hashed.
3. Distributions are typed discriminated unions (`LogNormal`, `Gamma`, `Uniform`, `Poisson`, `Empirical`, ...),
   never free-form strings.
4. Probabilities in [0, 1]; mixtures sum to 1 within 1e-9.
5. Sizes explicit per profile; no hidden defaults that differ between profiles.

## Key parameters added in the revision pass

### `suspicious_prevalence` (sweepable)

- Definition: the target share of **internal, normal-population accounts** that are members (any role) of at least
  one scenario instance over the simulation, including OOD families. Default **0.01** in DEV and RESEARCH.
- Mechanics: the generator converts the target into a per-family member quota
  `prevalence * n_accounts * family_weight / sum(enabled family weights)` and activates instances
  `0, 1, 2, ...` of each family (each with its randomly drawn member count) until the quota is met, up to that family's
  `max_instances` (decision 0002). If `max_instances` is reached before the quota, validation fails loudly rather than
  quietly producing a lower prevalence.
- The realized prevalence is computed and reported; a test checks it is within a tolerance of the target.
- It is intended to be swept for the distribution-shift experiments (see `docs/EXPERIMENTS.md`).

### Per-instance parameter distributions

Each family's config specifies **distributions** for instance-level parameters, not fixed values. Every instance draws
its own parameters from its own stream before generating events:

| Instance parameter | Example distribution (placeholder values, to be set in Milestone 1) |
|---|---|
| amount scale (median of the instance's amounts, minor units) | LogNormal |
| start offset | Uniform over the simulation |
| duration of each phase | Gamma per phase |
| counterparty count (fan degree, hop count) | shifted Poisson / Empirical |
| holding time | LogNormal |
| background-activity retention (share of normal behavior kept by members) | Beta |

Drawn values are stored in `ground_truth_networks`, so instance diversity is measurable. Milestone 1 validation reports,
per family, the coefficient of variation of each instance parameter across instances and fails if a family's instances
are near-identical (e.g. amount-scale CV below a configured floor).

### OOD family flag

`is_ood_family: bool` per family (replaces the earlier `split_role`). It affects only splits and label visibility
(decision 0009), never generation, so toggling it does not change the dataset content hash of event tables. (It does
change the `is_ood_family` column in label tables, by design.)

### Label knowledge block

```yaml
label_knowledge:
  regime: delayed               # 'delayed' (default) or 'oracle' (explicit ablation only)
  p_never_known: 0.10
  latency_days: {kind: LogNormal, median: 14.0, sigma: 0.75}
```

Validators (decision 0004): `regime: delayed` requires latency median >= 1 day and sigma >= 0.1; `Constant`
distributions are not allowed for latency; `p_never_known` in [0, 0.5].

### Logins block

Per archetype: login rate distribution, device count, probability of logging in from a new device or new IP, session
length. Digital-channel transactions must reference a successful login of the same account, device and IP within the
configured session window (validated invariant).

## Sketch (illustrative, not final)

```yaml
simulation: {seed: 42, days: 90, calendar_anchor: 2025-01-06}
population:
  n_persons: ...
  archetype_mix: {salary_worker: 0.35, student: 0.12, small_business: 0.08, ...}
suspicious_prevalence: 0.01
infrastructure:
  shared_ip_contexts: {household: ..., public_wifi: ..., corporate_nat: ..., mobile_cgnat: ...}
confounders:
  legit_high_volume_business: {share: ..., fan_in_degree: {kind: Gamma, ...}}
scenarios:
  - family: fan_in
    enabled: true
    weight: 1.0
    max_instances: 50
    is_ood_family: false
    instance_params:
      amount_scale_minor: {kind: LogNormal, median: ..., sigma: ...}
      n_sources: {kind: ShiftedPoisson, shift: 3, lam: ...}
    phases: {transition: ..., durations: ...}
  - family: cycle
    is_ood_family: true
    ...
label_knowledge: {regime: delayed, p_never_known: 0.10, latency_days: {kind: LogNormal, median: 14.0, sigma: 0.75}}
splits: {train_end_day: 60, val_end_day: 75, horizons_hours: [24, 168], ref_negative_fraction: 0.10}
```

## Safety rule embedded in the schema

Scenario parameters describe **pattern shape only**. No field means "stay below a detection threshold", references a
detector, or is chosen by searching against model scores. The generator never reads model outputs
(`docs/SECURITY_AND_ETHICS.md`). The structuring-like family is expressed as "many transfers of similar size clustered
in a configured amount band".

## Trade-offs

- No sweep syntax out of the box; a validated `--set key=value` override can be added later.
- Prevalence as a quota means the number of instances is a derived quantity; it is reported in metadata.
