"""EXP-M1-G non-triviality gates, run exactly as pre-registered in ``docs/EXPERIMENTS.md``.

Evaluation code: it may read ground truth (targets, cluster ids, archetypes for reporting). Features come
only from ``fcip.features.probe`` (store API). Probes use fixed hyperparameters; nothing is tuned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fcip.common.frames import to_df
from fcip.common.io import read_table
from fcip.common.rng import stream
from fcip.config.generator import GeneratorConfig
from fcip.evaluation.metrics import average_precision
from fcip.features.probe import context_features, single_account_feature_names, snapshot_features
from fcip.labels.oracle import OracleLabels
from fcip.temporal.splits import build_example_index, normal_account_ids, ref_pool_composition
from fcip.temporal.store import TemporalStore

G1_CEILING = 0.30
G2_CEILING = 0.60
G2_FLOOR_MULT = 3.0
G3_SAMPLE = 200_000
G3_BOOTSTRAP = 1000
G4_TOP_SHARE = 0.01
CLUSTER_WEAK = ("salary_worker", "traveler", "family", "spending_surge")
CLUSTER_DISTINCT = ("small_business", "hf_merchant", "student")


class _Prep:
    """log1p on features that are non-negative on TRAIN, then standardization fit on TRAIN."""

    def fit(self, x: np.ndarray) -> _Prep:
        self.nonneg = (x >= 0).all(axis=0)
        self.scaler = StandardScaler().fit(self._log(x))
        return self

    def _log(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float64).copy()
        out[:, self.nonneg] = np.log1p(out[:, self.nonneg])
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._log(x))


def _lr(x_tr: np.ndarray, y_tr: np.ndarray) -> tuple[_Prep, LogisticRegression]:
    prep = _Prep().fit(x_tr)
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
    m.fit(prep.transform(x_tr), y_tr)
    return prep, m


def _tree(x_tr: np.ndarray, y_tr: np.ndarray) -> DecisionTreeClassifier:
    seed = int(stream(0, "gate", "tree").integers(2**31 - 1))
    return DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=seed
    ).fit(x_tr, y_tr)


def _single_feature_aps(targets, x: np.ndarray, names: list[str]) -> dict[str, float]:
    return {
        n: max(average_precision(targets, x[:, j]), average_precision(targets, -x[:, j]))
        for j, n in enumerate(names)
    }


def _cluster_bootstrap(
    y: np.ndarray,
    clusters: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    oracle_targets,
    n_boot: int,
    seed_key: str,
) -> tuple[float, float, float]:
    """Delta AP (b - a) with a 95% cluster bootstrap CI; positive and negative clusters resampled separately."""
    from sklearn.metrics import average_precision_score

    rng = stream(0, "gate", seed_key)
    delta = average_precision(oracle_targets, s_b) - average_precision(oracle_targets, s_a)
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])
    has_pos = np.bincount(inv, weights=y.astype(float), minlength=len(uniq)) > 0
    pos_c, neg_c = np.flatnonzero(has_pos), np.flatnonzero(~has_pos)
    deltas = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(pos_c, len(pos_c)), rng.choice(neg_c, len(neg_c))])
        idx = np.concatenate([members[c] for c in pick])
        deltas.append(average_precision_score(y[idx], s_b[idx]) - average_precision_score(y[idx], s_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def run_seed(data_dir: Path, g3_sample: int = G3_SAMPLE, n_boot: int = G3_BOOTSTRAP,
             ref_pool_method: str | None = None) -> dict[str, Any]:
    """``ref_pool_method`` overrides the dataset's recorded split setting (used for the A2 decision)."""
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = GeneratorConfig.model_validate(meta["config"])
    if ref_pool_method is not None:
        cfg = cfg.model_copy(update={"splits": cfg.splits.model_copy(update={"ref_pool_method": ref_pool_method})})
    store = TemporalStore.from_dir(data_dir, meta["sim_end"])
    oracle = OracleLabels.from_dir(data_dir)
    pt = to_df(read_table("person_truth", data_dir))
    acc = to_df(read_table("accounts", data_dir)).filter(pl.col("account_kind") == "internal")
    arche = acc.join(pt, left_on="owner_person_id", right_on="person_id").select("account_id", "archetype")
    archetype_of = dict(zip(arche["account_id"].to_list(), arche["archetype"].to_list(), strict=True))

    idx = build_example_index(store, oracle, cfg, horizon_s=0, archetype_of=archetype_of)
    names = single_account_feature_names()
    train_pts, val_pts = idx.train("include_immature"), idx.val
    grid = snapshot_features(store, sorted(set(train_pts["day"].to_list()) | set(val_pts["day"].to_list())))
    tr = train_pts.join(grid, on=["account_id", "day"], how="left")
    va = val_pts.join(grid, on=["account_id", "day"], how="left")
    if (
        tr.select(names).null_count().sum_horizontal().item()
        or va.select(names).null_count().sum_horizontal().item()
    ):
        raise ValueError("missing features for some prediction points")
    x_tr, x_va = tr.select(names).to_numpy(), va.select(names).to_numpy()
    # Q-R5: probes are trained on ground-truth targets (strongest case, conservative for the ceilings)
    y_tr = oracle.targets("active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy()).y
    t_va = oracle.targets("active_phase", va["account_id"].to_numpy(), va["t"].to_numpy())
    pi = float(t_va.y.mean())

    g1 = _single_feature_aps(t_va, x_va, names)
    prep, lr = _lr(x_tr, y_tr)
    s_lr = lr.predict_proba(prep.transform(x_va))[:, 1]
    tree = _tree(x_tr, y_tr)
    s_tree = tree.predict_proba(x_va)[:, 1]
    ap_lr, ap_tree = average_precision(t_va, s_lr), average_precision(t_va, s_tree)

    # ---- G4: confounder share among top-scored false positives (report only)
    k = max(1, int(round(G4_TOP_SHARE * len(s_lr))))
    top = np.argsort(-s_lr, kind="stable")[:k]
    fp = va[top].filter(pl.Series(~t_va.y[top])).join(arche, on="account_id", how="left")
    g4 = fp.group_by("archetype").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))

    # ---- informational: separability by archetype cluster (review request)
    va_ar = va.join(arche, on="account_id", how="left")["archetype"].to_numpy()
    clusters_info: dict[str, dict[str, Any]] = {}
    for label, group in (("weak_on_aggregates", CLUSTER_WEAK), ("distinct_on_aggregates", CLUSTER_DISTINCT)):
        m = np.isin(va_ar, group)
        sub = oracle.targets("active_phase", va["account_id"].to_numpy()[m], va["t"].to_numpy()[m])
        if sub.n_positive == 0:
            clusters_info[label] = {"points": int(m.sum()), "positives": 0}
            continue
        sf = _single_feature_aps(sub, x_va[m], names)
        best = max(sf, key=lambda n: sf[n])
        clusters_info[label] = {
            "points": int(m.sum()),
            "positives": sub.n_positive,
            "prevalence": float(sub.y.mean()),
            "best_single_feature": best,
            "best_single_feature_ap": sf[best],
            "lr_ap": average_precision(sub, s_lr[m]),
        }

    # ---- G3: context contribution on uniform samples (report only)
    rng = stream(0, "gate", "g3-sample", int(meta["seed"]))
    tr_s = tr[np.sort(rng.choice(tr.height, min(g3_sample, tr.height), replace=False))]
    va_s = va[np.sort(rng.choice(va.height, min(g3_sample, va.height), replace=False))]
    c_tr = context_features(store, tr_s.select("account_id", "day"), grid)
    c_va = context_features(store, va_s.select("account_id", "day"), grid)
    ctx_names = c_tr.columns
    y_trs = oracle.targets("active_phase", tr_s["account_id"].to_numpy(), tr_s["t"].to_numpy()).y
    t_vas = oracle.targets("active_phase", va_s["account_id"].to_numpy(), va_s["t"].to_numpy())
    xs_tr, xs_va = tr_s.select(names).to_numpy(), va_s.select(names).to_numpy()
    p_a, m_a = _lr(xs_tr, y_trs)
    p_b, m_b = _lr(np.hstack([xs_tr, c_tr.to_numpy()]), y_trs)
    s_a = m_a.predict_proba(p_a.transform(xs_va))[:, 1]
    s_b = m_b.predict_proba(p_b.transform(np.hstack([xs_va, c_va.to_numpy()])))[:, 1]
    members = to_df(read_table("network_members", data_dir)).select("entity_id", "network_id")
    net_of = dict(zip(members["entity_id"].to_list(), members["network_id"].to_list(), strict=True))
    clus = np.asarray(
        [
            f"n{net_of[a]}" if (y and a in net_of) else f"a{a}"
            for a, y in zip(va_s["account_id"].to_list(), t_vas.y.tolist(), strict=True)
        ]
    )
    if t_vas.n_positive:
        d, lo, hi = _cluster_bootstrap(t_vas.y, clus, s_a, s_b, t_vas, n_boot, f"g3-boot-{meta['seed']}")
        g3 = {
            "sample_train": tr_s.height,
            "sample_val": va_s.height,
            "val_positives": t_vas.n_positive,
            "ap_static": average_precision(t_vas, s_a),
            "ap_static_plus_context": average_precision(t_vas, s_b),
            "delta_ap": d,
            "ci95": [lo, hi],
            "context_features": ctx_names,
        }
    else:
        g3 = {"sample_val": va_s.height, "val_positives": 0}

    comp = ref_pool_composition(normal_account_ids(store), idx.e_ref, archetype_of)
    best = max(g1, key=lambda n: g1[n])
    return {
        "data_dir": str(data_dir),
        "seed": meta["seed"],
        "profile": meta["profile"],
        "dataset_hash": meta["dataset_hash"],
        "realized_prevalence": meta["realized_prevalence"],
        "split_index_hash": idx.index_hash,
        "ref_pool_method": idx.ref_pool_method,
        "train_points": tr.height,
        "train_positives": int(y_tr.sum()),
        "val_points": va.height,
        "val_positives": t_va.n_positive,
        "pi": pi,
        "G1": {"best_feature": best, "best_ap": g1[best], "all": g1},
        "G2": {"ap_lr": ap_lr, "ap_tree": ap_tree, "floor": G2_FLOOR_MULT * pi},
        "G3": g3,
        "G4": g4.sort("share", descending=True).to_dicts(),
        "clusters": clusters_info,
        "ref_pool_composition": comp.to_dicts(),
    }


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered rule: a criterion holds if it holds for the mean over seeds and for at least
    (n - 1) of n seeds individually (4 of 5)."""
    n = len(results)
    need = n - 1

    def rule(values: list[bool], mean_ok: bool) -> bool:
        return mean_ok and sum(values) >= need

    g1 = [r["G1"]["best_ap"] for r in results]
    lr = [r["G2"]["ap_lr"] for r in results]
    tree = [r["G2"]["ap_tree"] for r in results]
    floor = [max(r["G2"]["ap_lr"], r["G2"]["ap_tree"]) >= r["G2"]["floor"] for r in results]
    mean_floor = float(np.mean([max(a, b) for a, b in zip(lr, tree, strict=True)])) >= float(
        np.mean([r["G2"]["floor"] for r in results])
    )
    return {
        "n_seeds": n,
        "G1_single_feature_ceiling": {
            "values": g1,
            "mean": float(np.mean(g1)),
            "threshold": G1_CEILING,
            "pass": rule([v <= G1_CEILING for v in g1], float(np.mean(g1)) <= G1_CEILING),
        },
        "G2a_probe_ceiling": {
            "lr": lr,
            "tree": tree,
            "threshold": G2_CEILING,
            "pass": rule(
                [a <= G2_CEILING and b <= G2_CEILING for a, b in zip(lr, tree, strict=True)],
                float(np.mean(lr)) <= G2_CEILING and float(np.mean(tree)) <= G2_CEILING,
            ),
        },
        "G2b_signal_floor": {"per_seed_pass": floor, "pass": rule(floor, mean_floor)},
    }
