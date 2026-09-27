"""Pydantic schema for the generator configuration (decision 0005).

All models forbid unknown keys and are frozen. Money is in int64 minor units of the single dataset
currency (S3); distributions over money therefore describe minor units.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fcip.common.taxonomy import Archetype, Family, Phase
from fcip.config.distributions import Dist, LogNormal, Uniform


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_probs(name: str, probs: dict[str, float]) -> None:
    if any(p < 0 for p in probs.values()) or abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must be non-negative and sum to 1, got {sum(probs.values())}")


class SimulationCfg(Strict):
    seed: int = Field(ge=0)
    days: int = Field(ge=14)
    calendar_anchor: dt.date

    @model_validator(mode="after")
    def _monday(self) -> SimulationCfg:
        if self.calendar_anchor.weekday() != 0:
            raise ValueError("calendar_anchor must be a Monday")
        return self


class CurrencyCfg(Strict):
    code: str = Field(min_length=3, max_length=3)
    minor_units_per_major: int = Field(gt=0)


class ExternalCfg(Strict):
    n_employers: int = Field(ge=1)
    n_billers: int = Field(ge=1)
    n_landlords: int = Field(ge=1)
    n_external_world: int = Field(ge=1)


class InfrastructureCfg(Strict):
    n_regions: int = Field(ge=1)
    n_atms_per_region: int = Field(ge=1)
    n_merchants_per_region: int = Field(ge=1)
    merchant_category_mix: dict[str, float]
    household_size: Dist
    devices_per_person: Dist
    p_desktop_device: float = Field(ge=0, le=1)
    p_household_shared_device: float = Field(ge=0, le=1)
    n_public_wifi_per_region: int = Field(ge=1)
    n_corporate_nat_per_region: int = Field(ge=1)
    n_cgnat_per_region: int = Field(ge=1)
    p_home_wifi_mobile: float = Field(ge=0, le=1)
    p_public_wifi: float = Field(ge=0, le=1)
    atm_amount_step_minor: int = Field(gt=0)
    atm_max_withdrawal_minor: int = Field(gt=0)
    session_seconds: Uniform
    session_window_seconds: int = Field(gt=0)
    p_new_account_in_sim: float = Field(ge=0, le=1)
    account_age_days: Dist

    @model_validator(mode="after")
    def _mix(self) -> InfrastructureCfg:
        _check_probs("merchant_category_mix", self.merchant_category_mix)
        if self.session_seconds.low < 5 or self.session_seconds.high > self.session_window_seconds:
            raise ValueError("session_seconds must lie within [5, session_window_seconds]")
        return self


class IncomeCfg(Strict):
    amount: Dist
    day_of_month: int = Field(ge=1, le=28)
    p_internal_employer: float = Field(ge=0, le=1)


class RentCfg(Strict):
    p_has_rent: float = Field(ge=0, le=1)
    amount: Dist
    day_of_month: int = Field(ge=1, le=28)
    p_internal_landlord: float = Field(ge=0, le=1)


class BillsCfg(Strict):
    n_bills: Dist
    amount: Dist


class CardCfg(Strict):
    daily_rate: Dist
    amount: Dist
    n_favorite_merchants: int = Field(ge=1)
    p_night: float = Field(ge=0, le=1)


class AtmCfg(Strict):
    weekly_rate: Dist
    amount: Dist


class P2PCfg(Strict):
    monthly_rate: Dist
    amount: Dist
    n_contacts: Dist
    p_same_region_contact: float = Field(ge=0, le=1)
    p_web: float = Field(ge=0, le=1)


class BusinessCfg(Strict):
    daily_customer_payments: Dist
    customer_amount: Dist
    p_internal_customer: float = Field(ge=0, le=1)
    weekly_supplier_payments: Dist
    supplier_amount: Dist
    daily_internal_payouts: Dist
    payout_amount: Dist
    is_employer: bool
    employer_weight: float = Field(gt=0)
    is_landlord: bool


class LoginCfg(Strict):
    daily_browse_rate: Dist
    p_failure: float = Field(ge=0, le=0.5)


class TravelCfg(Strict):
    trips: Dist
    trip_days: Dist


class SurgeCfg(Strict):
    duration_days: Dist
    rate_multiplier: Dist
    amount_multiplier: Dist


class DormancyCfg(Strict):
    p_dormant_gap: float = Field(ge=0, le=1)
    gap_days: Dist


class ArchetypeCfg(Strict):
    opening_balance: Dist
    overdraft_limit: Dist
    income: IncomeCfg | None = None
    rent: RentCfg | None = None
    bills: BillsCfg | None = None
    card: CardCfg | None = None
    atm: AtmCfg | None = None
    p2p: P2PCfg | None = None
    business: BusinessCfg | None = None
    logins: LoginCfg
    travel: TravelCfg | None = None
    surge: SurgeCfg | None = None
    dormancy: DormancyCfg | None = None
    has_workplace_ip: bool = False


class PopulationCfg(Strict):
    n_persons: int = Field(ge=10)
    archetype_mix: dict[Archetype, float]
    archetypes: dict[Archetype, ArchetypeCfg]

    @model_validator(mode="after")
    def _mix(self) -> PopulationCfg:
        _check_probs("archetype_mix", {k.value: v for k, v in self.archetype_mix.items()})
        missing = [a for a, p in self.archetype_mix.items() if p > 0 and a not in self.archetypes]
        if missing:
            raise ValueError(f"archetype_mix uses archetypes without config: {missing}")
        return self


class PhaseCfg(Strict):
    durations_hours: dict[Phase, Dist]
    p_repeat_round: float = Field(ge=0, lt=1)
    max_rounds: int = Field(ge=1)
    p_exit: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _phases(self) -> PhaseCfg:
        need = {Phase.SETUP, Phase.INFLOW, Phase.MOVEMENT, Phase.EXIT}
        missing = need - set(self.durations_hours)
        if missing:
            raise ValueError(f"durations_hours missing phases {sorted(missing)}")
        if Phase.HOLD in self.durations_hours or Phase.INACTIVE in self.durations_hours:
            raise ValueError(
                "HOLD duration comes from instance_params.holding_hours; INACTIVE has no duration"
            )
        return self


class InstanceParamsCfg(Strict):
    amount_scale_minor: Dist
    event_amount_sigma: float = Field(ge=0)
    fan_degree: Dist
    hops: Dist
    transfers_per_source: Dist
    holding_hours: Dist
    retention: Dist
    forward_fraction: Dist
    start_day: Dist
    p_victim_origin: float = Field(ge=0, le=1)
    amount_band_width: float = Field(gt=0, lt=1)
    dormant_gap_days: Dist


class FamilyCfg(Strict):
    enabled: bool
    weight: float = Field(gt=0)
    max_instances: int = Field(ge=1, le=4000)
    min_instances: int = Field(ge=0)
    is_ood_family: bool
    instance_params: InstanceParamsCfg
    phases: PhaseCfg

    @model_validator(mode="after")
    def _minmax(self) -> FamilyCfg:
        if self.min_instances > self.max_instances:
            raise ValueError("min_instances > max_instances")
        return self


class RecruitmentCfg(Strict):
    eligible_share: float = Field(gt=0, lt=1)
    max_shortfall_share: float = Field(ge=0, le=1)
    min_account_age_days_at_start: float = Field(ge=0)


class LabelKnowledgeCfg(Strict):
    regime: str
    p_never_known: float = Field(ge=0, le=0.5)
    latency_days: LogNormal

    @model_validator(mode="after")
    def _non_degenerate(self) -> LabelKnowledgeCfg:
        if self.regime not in ("delayed", "oracle"):
            raise ValueError(f"label_knowledge.regime must be 'delayed' or 'oracle', got {self.regime}")
        if self.regime == "delayed" and (self.latency_days.median < 1.0 or self.latency_days.sigma < 0.1):
            raise ValueError(
                "delayed label knowledge needs latency median >= 1 day and sigma >= 0.1 "
                "(use regime: oracle explicitly for zero latency)"
            )
        return self


class TrainingLabelsCfg(Strict):
    mode: str
    maturity_horizon_days: float = Field(gt=0)
    min_mature_train_days: float = Field(gt=0)

    @model_validator(mode="after")
    def _mode(self) -> TrainingLabelsCfg:
        if self.mode not in ("include_immature", "mature_only"):
            raise ValueError(f"training_labels.mode must be include_immature or mature_only, got {self.mode}")
        return self


class SplitsCfg(Strict):
    train_end_day: float = Field(gt=0)
    val_end_day: float = Field(gt=0)
    horizons_hours: list[float] = Field(min_length=1)
    ref_negative_fraction: float = Field(gt=0, lt=1)
    ref_pool_method: str = "none_cell_hash"  # or "stratified_archetype" (decision 0009, A2)

    @model_validator(mode="after")
    def _method(self) -> SplitsCfg:
        if self.ref_pool_method not in ("none_cell_hash", "stratified_archetype"):
            raise ValueError(f"unknown ref_pool_method {self.ref_pool_method}")
        return self

    @property
    def h_max_days(self) -> float:
        return max(self.horizons_hours) / 24.0


class ValidationCfg(Strict):
    min_instance_amount_cv: float = Field(default=0.1, ge=0)  # decision 0005: instances must not be templates
    min_instances_for_cv: int = Field(default=3, ge=2)


class GeneratorConfig(Strict):
    profile: str
    simulation: SimulationCfg
    currency: CurrencyCfg
    suspicious_prevalence: float = Field(gt=0, lt=0.5)
    high_risk_phases: list[Phase]
    external: ExternalCfg
    infrastructure: InfrastructureCfg
    population: PopulationCfg
    recruitment: RecruitmentCfg
    scenarios: dict[Family, FamilyCfg]
    label_knowledge: LabelKnowledgeCfg
    training_labels: TrainingLabelsCfg
    splits: SplitsCfg
    validation: ValidationCfg = ValidationCfg()

    @model_validator(mode="after")
    def _cross(self) -> GeneratorConfig:
        missing = set(Family) - set(self.scenarios)
        if missing:
            # The recruitment partition is defined over the whole family registry (decision 0002).
            raise ValueError(f"scenarios must list every family (enabled or not); missing {sorted(missing)}")
        days = self.simulation.days
        sp = self.splits
        if not 0 < sp.train_end_day < sp.val_end_day < days:
            raise ValueError("splits require 0 < train_end_day < val_end_day < days")
        h = sp.h_max_days
        for lo, hi, name in (
            (0, sp.train_end_day, "train"),
            (sp.train_end_day, sp.val_end_day, "val"),
            (sp.val_end_day, days, "test"),
        ):
            if h >= hi - lo:
                raise ValueError(f"H_max ({h} d) must be shorter than the {name} period ({hi - lo} d)")
        tl = self.training_labels
        if tl.mode == "mature_only":
            window = sp.train_end_day - h - tl.maturity_horizon_days
            if window < tl.min_mature_train_days:
                raise ValueError(
                    f"mature_only leaves {window:.1f} days of training prediction times "
                    f"(< min_mature_train_days={tl.min_mature_train_days}); see decision 0009"
                )
        if Phase.EXIT not in self.high_risk_phases:
            raise ValueError("high_risk_phases must include EXIT (terminal events are EXIT events)")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
