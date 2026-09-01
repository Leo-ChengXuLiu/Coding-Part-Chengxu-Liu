#!/usr/bin/env python3
"""Leakage-safe two-stage BDT core for tc vs full-background classification.

Expected input: one row per event AFTER the frozen Cut1 selection.
The script trains three controls:
  A) an ordinary event-level BDT using kinematics and reconstructed b tags;
  B) a direct-full BDT using every safe numeric event/jet feature;
  C) a double-stage BDT using every safe raw feature plus out-of-fold
     jet-flavour scores from a first-stage b/c/light classifier trained on
     every safe paired jet feature.

No truth-flavour column is ever used by the second-stage model.
Validation chooses the final score threshold; the untouched test set is used
once for the final S/B/Z comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from xgboost import XGBClassifier


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FLAVOR_TO_ID = {"b": 0, "c": 1, "light": 2}
ID_TO_FLAVOR = {v: k for k, v in FLAVOR_TO_ID.items()}
BTAG_EFF = {"b": 0.70, "c": 0.10, "light": 0.01}

REQUIRED_COLUMNS = {
    "event_id",
    "process",
    "is_signal",
    "physics_weight",
    "hj_truth_flavor",
    "lj_truth_flavor",
}

# Preferred feature order.  Automatic discovery below admits every numeric
# column that passes the explicit truth/label/weight leakage guard.
# Preferred order for automatically discovered paired HJ/LJ features.  Any
# additional safe numeric hj_*/lj_* pair in the input is appended, so future
# parquet columns are available without another code change.
JET_FEATURE_SUFFIX_PREFERRED = [
    "mass",
    "energy",
    "pt",
    "abs_eta",
    "n_constituents",
    "charged_mult",
    "neutral_mult",
    "girth",
    "width",
    "tau1",
    "tau2",
    "tau3",
    "tau21",
    "tau32",
    "d2",
    "lead_const_frac",
    "charged_energy_frac",
    "neutral_energy_frac",
    "ptd",
    "jet_charge",
    "btag",
]

EVENT_FEATURE_PREFERRED = [
    "hj_mass",
    "lj_mass",
    "hj_energy",
    "lj_energy",
    "hj_pt",
    "lj_pt",
    "hj_eta",
    "lj_eta",
    "hj_abs_eta",
    "lj_abs_eta",
    "hj_n_constituents",
    "lj_n_constituents",
    "hj_charged_mult",
    "lj_charged_mult",
    "hj_neutral_mult",
    "lj_neutral_mult",
    "hj_girth",
    "lj_girth",
    "hj_width",
    "lj_width",
    "hj_tau1",
    "lj_tau1",
    "hj_tau2",
    "lj_tau2",
    "hj_tau3",
    "lj_tau3",
    "hj_tau21",
    "lj_tau21",
    "hj_tau32",
    "lj_tau32",
    "hj_d2",
    "lj_d2",
    "hj_lead_const_frac",
    "lj_lead_const_frac",
    "mjj",
    "delta_r_jj",
    "delta_r_hj_lj",
    "delta_phi_hj_lj",
    "delta_eta_hj_lj",
    "energy_asymmetry",
    "pt_asymmetry",
    "visible_energy",
    "thrust",
    "sphericity",
    "aplanarity",
    "hj_btag",
    "lj_btag",
]

# Raw constituent/substructure variables are reserved for the first-stage
# flavour learner.  The ordinary event BDT does not see them; otherwise the
# comparison would ask whether a BDT benefits from receiving a nonlinear
# transform of variables it already has.  A separate direct-full control model
# receives every safe raw event variable.
FLAVOR_SUBSTRUCTURE_SUFFIXES = {
    "n_constituents",
    "charged_mult",
    "neutral_mult",
    "girth",
    "width",
    "tau1",
    "tau2",
    "tau3",
    "tau21",
    "tau32",
    "d2",
    "lead_const_frac",
    "charged_energy_frac",
    "neutral_energy_frac",
    "ptd",
    "jet_charge",
}

FORBIDDEN_FEATURE_TOKENS = {
    "truth",
    "flavor",
    "flavour",
    "label",
    "target",
    "signal",
    "process",
    "weight",
    "event_id",
    "pdg",
    "hadron",
    "pass_cut",
    "split",
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260716
    split_seed: int = 20260723
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    n_folds: int = 5
    n_jobs: int = 8
    device: str = "cuda"
    bootstrap_replicates: int = 500
    reference_coefficient: float = 0.032
    optimization_target_z: float = 2.0
    threshold_min_raw_signal: int = 50
    threshold_min_raw_background: int = 20

    jet_n_estimators: int = 900
    jet_max_depth: int = 6
    jet_learning_rate: float = 0.03

    event_n_estimators: int = 800
    event_max_depth: int = 5
    event_learning_rate: float = 0.03


# -----------------------------------------------------------------------------
# I/O and validation
# -----------------------------------------------------------------------------


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
        return table.rename(columns={"HJ_btag": "hj_btag", "LJ_btag": "lj_btag"})
    if suffix in {".csv", ".csv.gz"} or path.name.endswith(".csv.gz"):
        table = pd.read_csv(path)
        return table.rename(columns={"HJ_btag": "hj_btag", "LJ_btag": "lj_btag"})
    raise ValueError(f"Unsupported input format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in df.columns), None)


def validate_cut1_contract(df: pd.DataFrame, trust_cut1_input: bool) -> str:
    njet_col = find_column(df, ("Njet", "njet", "n_jet"))
    nlepton_col = find_column(df, ("Nlepton", "nlepton", "n_lepton"))

    require_columns(df, {"mjj"})
    bad_mjj = df["mjj"].isna() | (df["mjj"] <= 8000.0)
    if bad_mjj.any():
        examples = df.loc[bad_mjj, ["event_id", "mjj"]].head().to_dict("records")
        raise ValueError(f"Input violates Cut1 mjj > 8000 GeV; examples: {examples}")

    if njet_col is not None and nlepton_col is not None:
        bad = (pd.to_numeric(df[njet_col], errors="coerce") != 2) | (
            pd.to_numeric(df[nlepton_col], errors="coerce") != 0
        )
        if bad.any():
            examples = df.loc[bad, ["event_id", njet_col, nlepton_col]].head().to_dict("records")
            raise ValueError(f"Input violates Cut1 Njet == 2 and Nlepton == 0; examples: {examples}")
        return "verified_from_columns"

    if not trust_cut1_input:
        raise ValueError(
            "Cannot verify the Cut1 contract because Njet/Nlepton columns are absent. "
            "Regenerate the parquet with those columns, or explicitly pass "
            "--trust-cut1-input for a provenance-validated legacy table."
        )
    return "trusted_legacy_table_mjj_verified"


def validate_input(df: pd.DataFrame, trust_cut1_input: bool) -> str:
    require_columns(df, REQUIRED_COLUMNS)

    if df.empty:
        raise ValueError("Input table is empty")
    if df["event_id"].duplicated().any():
        dup = df.loc[df["event_id"].duplicated(), "event_id"].head().tolist()
        raise ValueError(f"Expected one row per Cut1 event; duplicate event_id examples: {dup}")
    if not set(df["is_signal"].dropna().unique()).issubset({0, 1, False, True}):
        raise ValueError("is_signal must be binary 0/1")
    if (df["physics_weight"] <= 0).any():
        raise ValueError("physics_weight must be positive")

    for col in ("hj_truth_flavor", "lj_truth_flavor"):
        bad = sorted(set(df[col].astype(str)) - set(FLAVOR_TO_ID))
        if bad:
            raise ValueError(f"{col} contains unsupported labels: {bad}")

    return validate_cut1_contract(df, trust_cut1_input)


def feature_is_safe(name: str) -> bool:
    lower = name.lower()
    return not any(token in lower for token in FORBIDDEN_FEATURE_TOKENS)


def numeric_safe_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and feature_is_safe(col)
    ]


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    safe_numeric = numeric_safe_columns(df)
    paired_suffixes = {
        col[3:]
        for col in safe_numeric
        if col.startswith("hj_") and f"lj_{col[3:]}" in safe_numeric
    }
    jet_features = [suffix for suffix in JET_FEATURE_SUFFIX_PREFERRED if suffix in paired_suffixes]
    jet_features.extend(sorted(paired_suffixes - set(jet_features)))

    full_event_features = [col for col in EVENT_FEATURE_PREFERRED if col in safe_numeric]
    full_event_features.extend(sorted(set(safe_numeric) - set(full_event_features)))

    ordinary_event_features = []
    for col in full_event_features:
        if col.startswith(("hj_", "lj_")) and col[3:] in FLAVOR_SUBSTRUCTURE_SUFFIXES:
            continue
        ordinary_event_features.append(col)

    if len(jet_features) < 3:
        raise ValueError(
            "Fewer than three common HJ/LJ jet features were found. "
            f"Available candidate suffixes: {jet_features}"
        )
    if len(ordinary_event_features) < 5:
        raise ValueError(
            "Fewer than five safe event features were found. "
            f"Available candidates: {ordinary_event_features}"
        )
    return jet_features, ordinary_event_features, full_event_features


def clean_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # Median values must be fit on training data in a production pipeline. Here
    # XGBoost can natively route missing values, so we deliberately keep NaNs.
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Event splitting: process-wise and event-safe
# -----------------------------------------------------------------------------


def stable_event_uniform(
    process: str,
    event_id: str,
    seed: int,
    namespace: str,
) -> float:
    payload = f"{namespace}:{seed}:{process}:{event_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def stable_event_bucket(
    process: str,
    event_id: str,
    seed: int,
    namespace: str,
    buckets: int,
) -> int:
    if buckets < 1:
        raise ValueError("buckets must be positive")
    payload = f"{namespace}:{seed}:{process}:{event_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % buckets


def assign_event_splits(df: pd.DataFrame, cfg: Config) -> pd.Series:
    if not math.isclose(
        cfg.train_fraction + cfg.validation_fraction + cfg.test_fraction,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("Split fractions must sum to 1")

    split = pd.Series(index=df.index, dtype="object")
    train_boundary = cfg.train_fraction
    validation_boundary = cfg.train_fraction + cfg.validation_fraction

    for process, group in df.groupby("process", sort=True):
        n = len(group)
        if n < 10:
            raise ValueError(f"Process {process!r} has too few Cut1 events: {n}")
        uniforms = np.fromiter(
            (
                stable_event_uniform(
                    str(process),
                    str(event_id),
                    cfg.split_seed,
                    "tc_bdt_global_split_v1",
                )
                for event_id in group["event_id"]
            ),
            dtype=np.float64,
            count=n,
        )
        labels = np.full(n, "test", dtype=object)
        labels[uniforms < validation_boundary] = "validation"
        labels[uniforms < train_boundary] = "train"
        split.loc[group.index] = labels

        observed = set(labels)
        if observed != {"train", "validation", "test"}:
            raise ValueError(
                f"Process {process!r} is missing a stable split: observed={sorted(observed)}"
            )

    if split.isna().any():
        raise AssertionError("Some events were not assigned a split")
    return split


def assert_disjoint_event_splits(df: pd.DataFrame) -> None:
    sets = {
        name: set(df.loc[df["split"] == name, "event_id"])
        for name in ("train", "validation", "test")
    }
    if sets["train"] & sets["validation"]:
        raise AssertionError("Train/validation event overlap")
    if sets["train"] & sets["test"]:
        raise AssertionError("Train/test event overlap")
    if sets["validation"] & sets["test"]:
        raise AssertionError("Validation/test event overlap")


# -----------------------------------------------------------------------------
# First stage: b/c/light jet classifier with OOF scores
# -----------------------------------------------------------------------------


def build_jet_table(events: pd.DataFrame, jet_feature_suffixes: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for role in ("hj", "lj"):
        role_cols = [f"{role}_{suffix}" for suffix in jet_feature_suffixes]
        part = events[["event_id", "process", "split", f"{role}_truth_flavor", *role_cols]].copy()
        part = part.rename(
            columns={
                f"{role}_truth_flavor": "truth_flavor",
                **{f"{role}_{suffix}": suffix for suffix in jet_feature_suffixes},
            }
        )
        part["jet_role"] = role
        frames.append(part)

    jets = pd.concat(frames, ignore_index=True)
    jets["flavor_id"] = jets["truth_flavor"].map(FLAVOR_TO_ID).astype(np.int8)
    return jets


def flavor_process_balanced_weights(jets: pd.DataFrame) -> np.ndarray:
    # Equal total weight for each non-empty (truth flavour, source process) cell.
    cell_size = jets.groupby(["truth_flavor", "process"])["event_id"].transform("size")
    weights = 1.0 / cell_size.to_numpy(dtype=float)
    # Then equalize the three flavour totals.
    flavor_totals = pd.Series(weights).groupby(jets["truth_flavor"].to_numpy()).transform("sum")
    weights = weights / flavor_totals.to_numpy(dtype=float)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def make_jet_model(cfg: Config, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=cfg.jet_n_estimators,
        max_depth=cfg.jet_max_depth,
        learning_rate=cfg.jet_learning_rate,
        min_child_weight=1.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        reg_alpha=0.0,
        tree_method="hist",
        device=cfg.device,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=cfg.n_jobs,
        verbosity=0,
    )


def choose_group_splitter(train_jets: pd.DataFrame, cfg: Config):
    combined = train_jets["truth_flavor"].astype(str) + "|" + train_jets["process"].astype(str)
    min_cell = int(combined.value_counts().min())
    if min_cell >= cfg.n_folds:
        return StratifiedGroupKFold(
            n_splits=cfg.n_folds,
            shuffle=True,
            random_state=cfg.seed,
        ), combined
    return GroupKFold(n_splits=cfg.n_folds), train_jets["flavor_id"]


def first_stage_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    clipped = np.clip(probabilities.astype(float), 1e-12, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    predicted = clipped.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y_true, predicted), 1)

    one_hot = np.eye(3, dtype=float)[y_true]
    confidence = clipped.max(axis=1)
    correct = (predicted == y_true).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))

    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                clipped,
                labels=np.arange(3),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auc = math.nan

    per_class_auc: dict[str, float] = {}
    for class_id in range(3):
        binary_target = (y_true == class_id).astype(int)
        try:
            per_class_auc[ID_TO_FLAVOR[class_id]] = float(
                roc_auc_score(binary_target, clipped[:, class_id])
            )
        except ValueError:
            per_class_auc[ID_TO_FLAVOR[class_id]] = math.nan

    return {
        "n_jets": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "macro_ovr_auc": macro_auc,
        "per_class_ovr_auc": per_class_auc,
        "multiclass_logloss": float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean()),
        "multiclass_brier": float(np.square(clipped - one_hot).sum(axis=1).mean()),
        "expected_calibration_error_10bin": ece,
        "confusion_matrix_rows_truth_cols_prediction": confusion.tolist(),
        "class_order": [ID_TO_FLAVOR[i] for i in range(3)],
    }


def fit_first_stage(
    events: pd.DataFrame,
    jet_feature_suffixes: Sequence[str],
    cfg: Config,
    model_dir: Path,
) -> tuple[pd.DataFrame, list[XGBClassifier], dict[str, object], pd.DataFrame]:
    jets = build_jet_table(events, jet_feature_suffixes)
    feature_cols = list(jet_feature_suffixes)

    train_jets = jets.loc[jets["split"] == "train"].copy()
    validation_jets = jets.loc[jets["split"] == "validation"].copy()
    test_jets = jets.loc[jets["split"] == "test"].copy()

    X_train = clean_numeric_frame(train_jets, feature_cols)
    y_train = train_jets["flavor_id"].to_numpy()
    groups = train_jets["event_id"].astype(str).to_numpy()
    splitter, split_y = choose_group_splitter(train_jets, cfg)

    oof = np.full((len(train_jets), 3), np.nan, dtype=np.float32)
    validation_pred = np.zeros((len(validation_jets), 3), dtype=np.float64)
    test_pred = np.zeros((len(test_jets), 3), dtype=np.float64)
    models: list[XGBClassifier] = []

    model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(X_train, split_y, groups=groups)):
        fit_groups = set(groups[fit_idx])
        hold_groups = set(groups[hold_idx])
        if fit_groups & hold_groups:
            raise AssertionError(f"First-stage fold {fold} has event leakage")

        fit_jets = train_jets.iloc[fit_idx]
        fit_weights = flavor_process_balanced_weights(fit_jets)
        model = make_jet_model(cfg, cfg.seed + fold)
        model.fit(X_train.iloc[fit_idx], y_train[fit_idx], sample_weight=fit_weights)

        oof[hold_idx] = model.predict_proba(X_train.iloc[hold_idx])
        if len(validation_jets):
            validation_pred += model.predict_proba(clean_numeric_frame(validation_jets, feature_cols))
        if len(test_jets):
            test_pred += model.predict_proba(clean_numeric_frame(test_jets, feature_cols))

        model.save_model(model_dir / f"jet_flavor_fold_{fold}.json")
        models.append(model)

    if np.isnan(oof).any():
        raise AssertionError("OOF first-stage predictions contain NaNs")

    validation_pred /= len(models)
    test_pred /= len(models)

    scored_parts: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "score_semantics": (
            "Normalized XGBoost flavour-like scores under balanced training priors; "
            "not calibrated physical flavour probabilities."
        )
    }
    for part, pred in (
        (train_jets, oof),
        (validation_jets, validation_pred),
        (test_jets, test_pred),
    ):
        out = part[["event_id", "jet_role", "split"]].copy()
        out["p_b"] = pred[:, FLAVOR_TO_ID["b"]]
        out["p_c"] = pred[:, FLAVOR_TO_ID["c"]]
        out["p_light"] = pred[:, FLAVOR_TO_ID["light"]]
        scored_parts.append(out)
        split_name = str(part["split"].iloc[0])
        metrics[split_name] = first_stage_metrics(part["flavor_id"].to_numpy(), pred)

    scored_jets = pd.concat(scored_parts, ignore_index=True)
    score_sum = scored_jets[["p_b", "p_c", "p_light"]].sum(axis=1)
    if not np.allclose(score_sum, 1.0, atol=2e-4):
        raise AssertionError("First-stage probabilities do not sum to one")

    fold_importances = np.vstack([model.feature_importances_ for model in models])
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": fold_importances.mean(axis=0),
            "importance_std": fold_importances.std(axis=0),
        }
    )
    for fold in range(fold_importances.shape[0]):
        importance[f"fold_{fold}"] = fold_importances[fold]
    importance = importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return scored_jets, models, metrics, importance


def attach_jet_scores(events: pd.DataFrame, scored_jets: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    for role in ("hj", "lj"):
        part = scored_jets.loc[scored_jets["jet_role"] == role, ["event_id", "p_b", "p_c", "p_light"]].copy()
        part = part.rename(
            columns={
                "p_b": f"{role}_p_b",
                "p_c": f"{role}_p_c",
                "p_light": f"{role}_p_light",
            }
        )
        out = out.merge(part, on="event_id", how="left", validate="one_to_one")

    score_cols = [
        "hj_p_b",
        "hj_p_c",
        "hj_p_light",
        "lj_p_b",
        "lj_p_c",
        "lj_p_light",
    ]
    if out[score_cols].isna().any().any():
        raise AssertionError("Missing first-stage scores after event merge")

    eps = 1e-8
    tc_like_score = np.clip(out["hj_p_b"] * out["lj_p_c"], eps, 1.0 - eps)
    out["bc_log_likelihood"] = np.log(tc_like_score / (1.0 - tc_like_score))
    return out


# -----------------------------------------------------------------------------
# Second stage: event BDT and physics-weighted evaluation
# -----------------------------------------------------------------------------


def class_balanced_physics_weights(events: pd.DataFrame) -> np.ndarray:
    y = events["is_signal"].astype(int).to_numpy()
    base = events["physics_weight"].to_numpy(dtype=float)
    weights = np.zeros_like(base)
    for label in (0, 1):
        mask = y == label
        total = base[mask].sum()
        if total <= 0:
            raise ValueError(f"Class {label} has zero total physics weight")
        weights[mask] = base[mask] / total
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def make_event_model(cfg: Config, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=cfg.event_n_estimators,
        max_depth=cfg.event_max_depth,
        learning_rate=cfg.event_learning_rate,
        min_child_weight=2.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        reg_alpha=0.0,
        tree_method="hist",
        device=cfg.device,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=cfg.n_jobs,
        verbosity=0,
    )


def fit_event_model(
    events: pd.DataFrame,
    features: Sequence[str],
    cfg: Config,
    seed_offset: int,
) -> XGBClassifier:
    train = events.loc[events["split"] == "train"]
    model = make_event_model(cfg, cfg.seed + seed_offset)
    model.fit(
        clean_numeric_frame(train, features),
        train["is_signal"].astype(int).to_numpy(),
        sample_weight=class_balanced_physics_weights(train),
    )
    return model


def add_split_extrapolation_weights(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    totals = out.groupby("process")["physics_weight"].sum()
    split_totals = out.groupby(["split", "process"])["physics_weight"].sum()

    factors: dict[tuple[str, str], float] = {}
    for (split, process), subtotal in split_totals.items():
        if subtotal <= 0:
            raise ValueError(f"Zero split weight for {split}/{process}")
        factors[(split, process)] = float(totals.loc[process] / subtotal)

    out["eval_weight"] = [
        w * factors[(split, process)]
        for w, split, process in zip(out["physics_weight"], out["split"], out["process"])
    ]
    return out


def significance(s: float, b: float) -> float:
    if s <= 0 or s + b <= 0:
        return 0.0
    return float(s / math.sqrt(s + b))


def coefficient_reach(s_reference: float, b: float, c_reference: float, target_z: float) -> float:
    """Solve Z=target_z for S(C)=S(C_ref)*(C/C_ref)^2."""
    if s_reference <= 0 or c_reference <= 0 or target_z <= 0 or b < 0:
        return math.inf
    scale = (
        target_z**2 + target_z * math.sqrt(target_z**2 + 4.0 * b)
    ) / (2.0 * s_reference)
    return float(c_reference * math.sqrt(scale))


def yields_at_threshold(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> tuple[float, float, float, float]:
    selected = df[score_col].to_numpy() >= threshold
    y = df["is_signal"].astype(int).to_numpy()
    w = df["eval_weight"].to_numpy(dtype=float)
    s = float(w[selected & (y == 1)].sum())
    b = float(w[selected & (y == 0)].sum())
    return s, b, s / b if b > 0 else math.inf, significance(s, b)


def optimize_threshold(
    validation: pd.DataFrame,
    score_col: str,
    reference_coefficient: float,
    target_z: float,
    min_raw_signal: int,
    min_raw_background: int,
    min_signal_efficiency: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    scores = validation[score_col].to_numpy(dtype=float)
    quantiles = np.linspace(0.0, 1.0, 1001)
    thresholds = np.unique(np.quantile(scores, quantiles))

    total_signal = validation.loc[validation["is_signal"] == 1, "eval_weight"].sum()
    labels = validation["is_signal"].astype(int).to_numpy()
    rows = []
    for threshold in thresholds:
        selected = scores >= threshold
        s, b, s_over_b, z = yields_at_threshold(validation, score_col, float(threshold))
        signal_eff = float(s / total_signal) if total_signal > 0 else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "S": s,
                "B": b,
                "S_over_B": s_over_b,
                "Z": z,
                "signal_efficiency": signal_eff,
                "raw_signal_events": int(np.count_nonzero(selected & (labels == 1))),
                "raw_background_events": int(np.count_nonzero(selected & (labels == 0))),
                "coefficient_reach": coefficient_reach(
                    s,
                    b,
                    reference_coefficient,
                    target_z,
                ),
            }
        )

    scan = pd.DataFrame(rows)
    eligible = scan.loc[
        (scan["signal_efficiency"] >= min_signal_efficiency)
        & (scan["raw_signal_events"] >= min_raw_signal)
        & (scan["raw_background_events"] >= min_raw_background)
    ]
    if eligible.empty:
        raise RuntimeError("No event-BDT threshold passes the minimum signal-efficiency requirement")
    best = eligible.sort_values(
        ["coefficient_reach", "Z", "S_over_B"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"]), scan


def manual_baseline(df: pd.DataFrame) -> dict[str, float | str]:
    require_columns(df, {"hj_mass", "lj_mass"})
    mass_mask = (df["hj_mass"] > 150.0) & (df["hj_mass"] < 200.0) & (df["lj_mass"] < 75.0)
    if {"hj_btag", "lj_btag"}.issubset(df.columns):
        selected_weight = (
            mass_mask
            & (pd.to_numeric(df["hj_btag"], errors="coerce") > 0.5)
            & (pd.to_numeric(df["lj_btag"], errors="coerce") <= 0.5)
        ).to_numpy(dtype=float)
        mode = "observed_reconstructed_btag_flags"
    else:
        p_hj = df["hj_truth_flavor"].map(BTAG_EFF).to_numpy(dtype=float)
        p_lj = df["lj_truth_flavor"].map(BTAG_EFF).to_numpy(dtype=float)
        selected_weight = mass_mask.to_numpy(dtype=float) * p_hj * (1.0 - p_lj)
        mode = "analytic_expected_btag_fallback"
    y = df["is_signal"].astype(int).to_numpy()
    w = df["eval_weight"].to_numpy(dtype=float)
    s = float((w * selected_weight * (y == 1)).sum())
    b = float((w * selected_weight * (y == 0)).sum())
    return {
        "mode": mode,
        "S": s,
        "B": b,
        "S_over_B": s / b if b > 0 else math.inf,
        "Z": significance(s, b),
    }


def cut3_mask(df: pd.DataFrame) -> np.ndarray:
    require_columns(df, {"hj_mass", "lj_mass"})
    return (
        (df["hj_mass"].to_numpy(dtype=float) > 150.0)
        & (df["hj_mass"].to_numpy(dtype=float) < 200.0)
        & (df["lj_mass"].to_numpy(dtype=float) < 75.0)
    )


def yields_from_mask(df: pd.DataFrame, selected: np.ndarray) -> tuple[float, float, float, float]:
    y = df["is_signal"].astype(int).to_numpy()
    w = df["eval_weight"].to_numpy(dtype=float)
    s = float(w[selected & (y == 1)].sum())
    b = float(w[selected & (y == 0)].sum())
    return s, b, s / b if b > 0 else math.inf, significance(s, b)


def optimize_cut3_charm_threshold(
    validation: pd.DataFrame,
    reference_coefficient: float,
    target_z: float,
    min_raw_signal: int,
    min_raw_background: int,
    min_cut3_signal_efficiency: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Optimize a transparent LJ charm-score cut after the frozen Cut3 mask."""
    base_mask = cut3_mask(validation)
    scores = validation["lj_p_c"].to_numpy(dtype=float)
    thresholds = np.unique(np.quantile(scores[base_mask], np.linspace(0.0, 1.0, 501)))
    base_signal = yields_from_mask(validation, base_mask)[0]
    labels = validation["is_signal"].astype(int).to_numpy()
    rows = []
    for threshold in thresholds:
        selected = base_mask & (scores >= threshold)
        s, b, s_over_b, z = yields_from_mask(validation, selected)
        signal_efficiency = s / base_signal if base_signal > 0 else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "S": s,
                "B": b,
                "S_over_B": s_over_b,
                "Z": z,
                "cut3_relative_signal_efficiency": float(signal_efficiency),
                "raw_signal_events": int(np.count_nonzero(selected & (labels == 1))),
                "raw_background_events": int(np.count_nonzero(selected & (labels == 0))),
                "coefficient_reach": coefficient_reach(s, b, reference_coefficient, target_z),
            }
        )
    scan = pd.DataFrame(rows)
    eligible = scan.loc[
        (scan["cut3_relative_signal_efficiency"] >= min_cut3_signal_efficiency)
        & (scan["raw_signal_events"] >= min_raw_signal)
        & (scan["raw_background_events"] >= min_raw_background)
    ]
    if eligible.empty:
        raise RuntimeError("No Cut3+charm threshold passes the signal-efficiency requirement")
    best = eligible.sort_values(
        ["coefficient_reach", "Z", "S_over_B"], ascending=[True, False, False]
    ).iloc[0]
    return float(best["threshold"]), scan


def process_yields(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    selected = df[score_col] >= threshold
    rows = []
    for process, group in df.loc[selected].groupby("process"):
        rows.append(
            {
                "process": process,
                "yield": float(group["eval_weight"].sum()),
                "raw_selected": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("yield", ascending=False)


def bootstrap_z(
    test: pd.DataFrame,
    score_col: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    by_process = {process: group for process, group in test.groupby("process")}
    values = np.empty(replicates, dtype=float)

    for i in range(replicates):
        sampled_parts = []
        for group in by_process.values():
            positions = rng.integers(0, len(group), size=len(group))
            sampled_parts.append(group.iloc[positions])
        sample = pd.concat(sampled_parts, ignore_index=True)
        _, _, _, values[i] = yields_at_threshold(sample, score_col, threshold)

    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return {
        "median": float(q50),
        "q16": float(q16),
        "q84": float(q84),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Cut1 event feature table (.csv/.parquet)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=Config.device,
        help="XGBoost training device; CUDA is preferred for the full production sample",
    )
    parser.add_argument("--bootstrap", type=int, default=Config.bootstrap_replicates)
    parser.add_argument("--reference-coefficient", type=float, default=Config.reference_coefficient)
    parser.add_argument("--target-z", type=float, default=Config.optimization_target_z)
    parser.add_argument("--threshold-min-raw-signal", type=int, default=Config.threshold_min_raw_signal)
    parser.add_argument(
        "--threshold-min-raw-background",
        type=int,
        default=Config.threshold_min_raw_background,
    )
    parser.add_argument("--jet-n-estimators", type=int, default=Config.jet_n_estimators)
    parser.add_argument("--jet-max-depth", type=int, default=Config.jet_max_depth)
    parser.add_argument("--jet-learning-rate", type=float, default=Config.jet_learning_rate)
    parser.add_argument("--event-n-estimators", type=int, default=Config.event_n_estimators)
    parser.add_argument("--event-max-depth", type=int, default=Config.event_max_depth)
    parser.add_argument("--event-learning-rate", type=float, default=Config.event_learning_rate)
    parser.add_argument(
        "--trust-cut1-input",
        action="store_true",
        help="Accept a legacy Cut1 parquet without Njet/Nlepton columns after verifying mjj > 8000",
    )
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        n_jobs=args.n_jobs,
        device=args.device,
        bootstrap_replicates=args.bootstrap,
        reference_coefficient=args.reference_coefficient,
        optimization_target_z=args.target_z,
        threshold_min_raw_signal=args.threshold_min_raw_signal,
        threshold_min_raw_background=args.threshold_min_raw_background,
        jet_n_estimators=args.jet_n_estimators,
        jet_max_depth=args.jet_max_depth,
        jet_learning_rate=args.jet_learning_rate,
        event_n_estimators=args.event_n_estimators,
        event_max_depth=args.event_max_depth,
        event_learning_rate=args.event_learning_rate,
    )
    out_dir = args.output
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    events = read_table(args.input)
    cut1_contract = validate_input(events, args.trust_cut1_input)
    jet_features, ordinary_features, direct_full_features = select_features(events)

    events = events.copy()
    events["event_id"] = events["event_id"].astype(str)
    events["split"] = assign_event_splits(events, cfg)
    assert_disjoint_event_splits(events)

    # First-stage OOF scores.
    scored_jets, _, jet_stage_metrics, jet_importance = fit_first_stage(
        events, jet_features, cfg, model_dir
    )
    events = attach_jet_scores(events, scored_jets)
    events = add_split_extrapolation_weights(events)

    score_features = [
        "hj_p_b",
        "hj_p_c",
        "hj_p_light",
        "lj_p_b",
        "lj_p_c",
        "lj_p_light",
        "bc_log_likelihood",
    ]
    double_features = list(dict.fromkeys([*direct_full_features, *score_features]))

    for name in ordinary_features + direct_full_features + double_features:
        if not feature_is_safe(name) and name not in score_features:
            raise AssertionError(f"Unsafe second-stage feature selected: {name}")

    ordinary = fit_event_model(events, ordinary_features, cfg, seed_offset=100)
    direct_full = fit_event_model(events, direct_full_features, cfg, seed_offset=150)
    double = fit_event_model(events, double_features, cfg, seed_offset=200)
    ordinary.save_model(model_dir / "event_bdt_ordinary.json")
    direct_full.save_model(model_dir / "event_bdt_direct_full.json")
    double.save_model(model_dir / "event_bdt_double_stage.json")

    for split_name in ("train", "validation", "test"):
        mask = events["split"] == split_name
        events.loc[mask, "ordinary_score"] = ordinary.predict_proba(
            clean_numeric_frame(events.loc[mask], ordinary_features)
        )[:, 1]
        events.loc[mask, "direct_full_score"] = direct_full.predict_proba(
            clean_numeric_frame(events.loc[mask], direct_full_features)
        )[:, 1]
        events.loc[mask, "double_score"] = double.predict_proba(
            clean_numeric_frame(events.loc[mask], double_features)
        )[:, 1]

    validation = events.loc[events["split"] == "validation"].copy()
    test = events.loc[events["split"] == "test"].copy()

    ordinary_threshold, ordinary_scan = optimize_threshold(
        validation,
        "ordinary_score",
        cfg.reference_coefficient,
        cfg.optimization_target_z,
        cfg.threshold_min_raw_signal,
        cfg.threshold_min_raw_background,
    )
    direct_full_threshold, direct_full_scan = optimize_threshold(
        validation,
        "direct_full_score",
        cfg.reference_coefficient,
        cfg.optimization_target_z,
        cfg.threshold_min_raw_signal,
        cfg.threshold_min_raw_background,
    )
    double_threshold, double_scan = optimize_threshold(
        validation,
        "double_score",
        cfg.reference_coefficient,
        cfg.optimization_target_z,
        cfg.threshold_min_raw_signal,
        cfg.threshold_min_raw_background,
    )
    cut3_charm_threshold, cut3_charm_scan = optimize_cut3_charm_threshold(
        validation,
        cfg.reference_coefficient,
        cfg.optimization_target_z,
        cfg.threshold_min_raw_signal,
        cfg.threshold_min_raw_background,
    )

    ordinary_test = yields_at_threshold(test, "ordinary_score", ordinary_threshold)
    direct_full_test = yields_at_threshold(test, "direct_full_score", direct_full_threshold)
    double_test = yields_at_threshold(test, "double_score", double_threshold)
    manual_test = manual_baseline(test)
    cut3_test = yields_from_mask(test, cut3_mask(test))
    cut3_charm_test = yields_from_mask(
        test,
        cut3_mask(test) & (test["lj_p_c"].to_numpy(dtype=float) >= cut3_charm_threshold),
    )

    ordinary_auc = roc_auc_score(
        test["is_signal"].astype(int),
        test["ordinary_score"],
        sample_weight=test["eval_weight"],
    )
    direct_full_auc = roc_auc_score(
        test["is_signal"].astype(int),
        test["direct_full_score"],
        sample_weight=test["eval_weight"],
    )
    double_auc = roc_auc_score(
        test["is_signal"].astype(int),
        test["double_score"],
        sample_weight=test["eval_weight"],
    )

    ordinary_boot = bootstrap_z(
        test,
        "ordinary_score",
        ordinary_threshold,
        cfg.bootstrap_replicates,
        cfg.seed + 1000,
    )
    direct_full_boot = bootstrap_z(
        test,
        "direct_full_score",
        direct_full_threshold,
        cfg.bootstrap_replicates,
        cfg.seed + 1500,
    )
    double_boot = bootstrap_z(
        test,
        "double_score",
        double_threshold,
        cfg.bootstrap_replicates,
        cfg.seed + 2000,
    )

    summary = {
        "config": asdict(cfg),
        "input": str(args.input.resolve()),
        "cut1_contract": cut1_contract,
        "n_events_cut1": int(len(events)),
        "process_counts_cut1": {k: int(v) for k, v in events["process"].value_counts().items()},
        "split_counts": {k: int(v) for k, v in events["split"].value_counts().items()},
        "jet_features": jet_features,
        "first_stage_metrics": jet_stage_metrics,
        "ordinary_event_features": ordinary_features,
        "direct_full_event_features": direct_full_features,
        "double_stage_features": double_features,
        "manual_baseline_test": manual_test,
        "cut3_test": {
            "S": cut3_test[0],
            "B": cut3_test[1],
            "S_over_B": cut3_test[2],
            "Z": cut3_test[3],
            "coefficient_reach_Z2": coefficient_reach(
                cut3_test[0], cut3_test[1], cfg.reference_coefficient, 2.0
            ),
            "coefficient_reach_Z3": coefficient_reach(
                cut3_test[0], cut3_test[1], cfg.reference_coefficient, 3.0
            ),
        },
        "cut3_plus_learned_lj_charm_test": {
            "threshold_from_validation": cut3_charm_threshold,
            "S": cut3_charm_test[0],
            "B": cut3_charm_test[1],
            "S_over_B": cut3_charm_test[2],
            "Z": cut3_charm_test[3],
            "coefficient_reach_Z2": coefficient_reach(
                cut3_charm_test[0], cut3_charm_test[1], cfg.reference_coefficient, 2.0
            ),
            "coefficient_reach_Z3": coefficient_reach(
                cut3_charm_test[0], cut3_charm_test[1], cfg.reference_coefficient, 3.0
            ),
            "delta_Z_vs_cut3": float(cut3_charm_test[3] - cut3_test[3]),
        },
        "ordinary_event_bdt_test": {
            "threshold_from_validation": ordinary_threshold,
            "S": ordinary_test[0],
            "B": ordinary_test[1],
            "S_over_B": ordinary_test[2],
            "Z": ordinary_test[3],
            "weighted_AUC": float(ordinary_auc),
            "coefficient_reach_Z2": coefficient_reach(
                ordinary_test[0], ordinary_test[1], cfg.reference_coefficient, 2.0
            ),
            "coefficient_reach_Z3": coefficient_reach(
                ordinary_test[0], ordinary_test[1], cfg.reference_coefficient, 3.0
            ),
            "bootstrap_Z": ordinary_boot,
        },
        "direct_full_event_bdt_test": {
            "threshold_from_validation": direct_full_threshold,
            "S": direct_full_test[0],
            "B": direct_full_test[1],
            "S_over_B": direct_full_test[2],
            "Z": direct_full_test[3],
            "weighted_AUC": float(direct_full_auc),
            "coefficient_reach_Z2": coefficient_reach(
                direct_full_test[0], direct_full_test[1], cfg.reference_coefficient, 2.0
            ),
            "coefficient_reach_Z3": coefficient_reach(
                direct_full_test[0], direct_full_test[1], cfg.reference_coefficient, 3.0
            ),
            "bootstrap_Z": direct_full_boot,
        },
        "double_stage_bdt_test": {
            "threshold_from_validation": double_threshold,
            "S": double_test[0],
            "B": double_test[1],
            "S_over_B": double_test[2],
            "Z": double_test[3],
            "weighted_AUC": float(double_auc),
            "coefficient_reach_Z2": coefficient_reach(
                double_test[0], double_test[1], cfg.reference_coefficient, 2.0
            ),
            "coefficient_reach_Z3": coefficient_reach(
                double_test[0], double_test[1], cfg.reference_coefficient, 3.0
            ),
            "bootstrap_Z": double_boot,
        },
        "delta_Z_double_minus_ordinary": float(double_test[3] - ordinary_test[3]),
        "relative_delta_Z_double_vs_ordinary": float(
            (double_test[3] / ordinary_test[3] - 1.0) if ordinary_test[3] > 0 else math.inf
        ),
        "delta_Z_double_minus_direct_full": float(double_test[3] - direct_full_test[3]),
        "relative_delta_Z_double_vs_direct_full": float(
            (double_test[3] / direct_full_test[3] - 1.0)
            if direct_full_test[3] > 0
            else math.inf
        ),
    }
    summary["manual_baseline_test"]["coefficient_reach_Z2"] = coefficient_reach(
        manual_test["S"], manual_test["B"], cfg.reference_coefficient, 2.0
    )
    summary["manual_baseline_test"]["coefficient_reach_Z3"] = coefficient_reach(
        manual_test["S"], manual_test["B"], cfg.reference_coefficient, 3.0
    )
    ordinary_cmin2 = summary["ordinary_event_bdt_test"]["coefficient_reach_Z2"]
    direct_full_cmin2 = summary["direct_full_event_bdt_test"]["coefficient_reach_Z2"]
    double_cmin2 = summary["double_stage_bdt_test"]["coefficient_reach_Z2"]
    summary["delta_Cmin_Z2_double_minus_ordinary"] = float(double_cmin2 - ordinary_cmin2)
    summary["delta_Cmin_Z2_double_minus_direct_full"] = float(
        double_cmin2 - direct_full_cmin2
    )
    summary["relative_delta_Cmin_Z2_double_vs_ordinary"] = float(
        double_cmin2 / ordinary_cmin2 - 1.0
    )
    summary["relative_delta_Cmin_Z2_double_vs_direct_full"] = float(
        double_cmin2 / direct_full_cmin2 - 1.0
    )

    events.to_csv(out_dir / "event_scores_and_splits.csv.gz", index=False)
    scored_jets.to_csv(out_dir / "jet_oof_scores.csv.gz", index=False)
    ordinary_scan.to_csv(out_dir / "ordinary_threshold_scan_validation.csv", index=False)
    direct_full_scan.to_csv(out_dir / "direct_full_threshold_scan_validation.csv", index=False)
    double_scan.to_csv(out_dir / "double_threshold_scan_validation.csv", index=False)
    cut3_charm_scan.to_csv(out_dir / "cut3_lj_charm_threshold_scan_validation.csv", index=False)
    process_yields(test, "ordinary_score", ordinary_threshold).to_csv(
        out_dir / "ordinary_test_process_yields.csv", index=False
    )
    process_yields(test, "direct_full_score", direct_full_threshold).to_csv(
        out_dir / "direct_full_test_process_yields.csv", index=False
    )
    process_yields(test, "double_score", double_threshold).to_csv(
        out_dir / "double_test_process_yields.csv", index=False
    )
    pd.DataFrame(
        {
            "feature": ordinary_features,
            "importance": ordinary.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        out_dir / "ordinary_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "feature": direct_full_features,
            "importance": direct_full.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        out_dir / "direct_full_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "feature": double_features,
            "importance": double.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        out_dir / "double_feature_importance.csv", index=False
    )
    jet_importance.to_csv(out_dir / "jet_feature_importance.csv", index=False)
    with open(out_dir / "jet_stage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(jet_stage_metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "jet_feature_suffixes": jet_features,
                "first_stage_score_semantics": jet_stage_metrics["score_semantics"],
                "ordinary_event_features": ordinary_features,
                "direct_full_event_features": direct_full_features,
                "double_stage_features": double_features,
            },
            f,
            indent=2,
        )
    joblib.dump(
        {
            "ordinary_threshold": ordinary_threshold,
            "direct_full_threshold": direct_full_threshold,
            "double_threshold": double_threshold,
            "cut3_lj_charm_threshold": cut3_charm_threshold,
        },
        out_dir / "selected_thresholds.joblib",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
