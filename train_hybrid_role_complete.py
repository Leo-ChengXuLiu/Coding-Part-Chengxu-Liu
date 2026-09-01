#!/usr/bin/env python3
"""Train leakage-safe CUDA BDT baselines on the production-v3 Cut1 sample.

The default pipeline compares four models on one frozen event split:

* event_only: contracted event-level observables only;
* direct_full: event observables plus truth-free HJ/LJ jet observables;
* double_stage: event observables plus the role-specific HJ-b and LJ-c
  out-of-fold scores;
* hybrid: direct_full inputs plus those two role-specific scores;
* double_stage_role_complete: double_stage plus an LJ-b rejection head and
  deterministic role-score combinations;
* hybrid_role_complete: hybrid plus the same role-complete flavor summary.

The first stage contains role-specific HJ b-vs-rest, LJ c-vs-rest, and LJ
b-vs-rest heads. The first two preserve the previous baseline while the third
explicitly represents the LJ no-b requirement used by the manual flavor cut.
A shared b/c/light classifier can be enabled explicitly as an ablation. Train
events receive strictly out-of-fold scores. Validation and test events are
scored by fold ensembles trained only on the training split. No truth label or
truth-proxy feature enters a stage-2 event classifier.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier


SCRIPT_PATH = Path(__file__).resolve()
RELEASE_ROOT = SCRIPT_PATH.parent
CORE_DIR = RELEASE_ROOT / "src"
sys.path.insert(0, str(CORE_DIR))
import double_stage_bdt_core as core  # noqa: E402


FLAVOR_TO_ID = core.FLAVOR_TO_ID
ROLE_NAMES = ("HJ", "LJ")
ROLE_TASKS = {
    "hj_b": {
        "role": "HJ",
        "target": "b",
        "column": "hj_role_p_b_oof",
        "seed_offset": 100,
    },
    "lj_c": {
        "role": "LJ",
        "target": "c",
        "column": "lj_role_p_c_oof",
        "seed_offset": 200,
    },
    "lj_b": {
        "role": "LJ",
        "target": "b",
        "column": "lj_role_p_b_oof",
        "seed_offset": 300,
    },
}
BASE_ROLE_SCORE_FEATURES = ["hj_role_p_b_oof", "lj_role_p_c_oof"]
ROLE_COMPLETE_SCORE_FEATURES = [
    *BASE_ROLE_SCORE_FEATURES,
    "lj_role_p_b_oof",
    "lj_role_c_over_b_log_score_oof",
    "tc_role_score_product_oof",
    "tc_role_score_contrast_oof",
]
ROLE_DIAGNOSTIC_SCORE_FEATURES = ["lj_role_not_b_oof"]
SPLITS = ("train", "validation", "test")
STAGE2_SEED_OFFSETS = {
    "event_only": 1000,
    "direct_full": 2000,
    "double_stage": 3000,
    "hybrid": 4000,
    "double_stage_role_complete": 5000,
    "hybrid_role_complete": 6000,
    "double_stage_shared": 7000,
    "hybrid_shared": 8000,
}
CARD_PIPELINE_MODEL_PARALLEL = True
STAGE1_BUNDLE_SCHEMA_VERSION = "tc_bdt_stage1_bundle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Root containing <process>/shard_*/{events,jets}.parquet.",
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=RELEASE_ROOT / "configs/BDT_FEATURES_DETECTOR.yaml",
    )
    parser.add_argument(
        "--microshard-manifest",
        type=Path,
        help=(
            "Optional generated-level microshard selection and split manifest. "
            "When provided, only listed source-event blocks are loaded."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260723,
        help="Fixed event-hash split seed; keep unchanged across subsets and models.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-shards-per-process", type=int)
    parser.add_argument("--jet-n-estimators", type=int, default=600)
    parser.add_argument("--jet-max-depth", type=int, default=6)
    parser.add_argument("--jet-learning-rate", type=float, default=0.035)
    parser.add_argument("--event-n-estimators", type=int, default=700)
    parser.add_argument("--event-max-depth", type=int, default=5)
    parser.add_argument("--event-learning-rate", type=float, default=0.035)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--reference-coefficient", type=float, default=0.032)
    parser.add_argument(
        "--physics-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply physical event weights after loading. Use the inverse retained "
            "shard fraction for a process-stratified fixed shard subset."
        ),
    )
    parser.add_argument("--threshold-min-raw-signal", type=int, default=50)
    parser.add_argument("--threshold-min-raw-background", type=int, default=20)
    parser.add_argument(
        "--mjj-min-gev",
        type=float,
        default=0.0,
        help="Additional training preselection applied after loading feature shards.",
    )
    parser.add_argument(
        "--include-shared-ablation",
        action="store_true",
        help=(
            "Also train a shared b/c/light OOF head and evaluate shared-score-only "
            "double-stage and hybrid ablations. The default models remain role-only."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Optional stage-2 model names to train. By default all available models "
            "are trained."
        ),
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Train and persist the shared OOF flavor stage, then stop.",
    )
    parser.add_argument(
        "--stage1-input",
        type=Path,
        help="Reuse a completed shared OOF flavor stage instead of retraining it.",
    )
    return parser.parse_args()


def read_contract(path: Path) -> tuple[list[str], list[str], list[str], str]:
    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    event_features = list(contract["event_bdt"]["features"])
    jet_features = list(contract["jet_flavor_stage"]["features"])
    first_stage_scope = str(
        contract["jet_flavor_stage"].get("scope", "truth_proxy_feasibility_only")
    )
    forbidden = list(contract["forbidden_patterns"])
    if len(event_features) != len(set(event_features)):
        raise ValueError("Duplicate event features in BDT_FEATURES.yaml")
    if len(jet_features) != len(set(jet_features)):
        raise ValueError("Duplicate jet features in BDT_FEATURES.yaml")
    return event_features, jet_features, forbidden, first_stage_scope


def discover_shards(root: Path, limit: int | None) -> list[Path]:
    shards: list[Path] = []
    for process_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        process_shards = sorted(process_dir.glob("shard_*"))
        if limit is not None:
            process_shards = process_shards[:limit]
        shards.extend(process_shards)
    if not shards:
        raise FileNotFoundError(f"No feature shards found below {root}")
    return shards


def microshard_read_ranges(
    manifest: dict[str, object] | None,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    if manifest is None:
        return {}
    microshard_size = int(manifest["microshard_size"])
    ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for process, process_payload in dict(manifest["processes"]).items():
        indices_by_shard: dict[str, list[int]] = {}
        for unit in dict(process_payload)["selected_units"]:
            row = dict(unit)
            indices_by_shard.setdefault(str(row["source_shard"]), []).append(
                int(row["microshard_index"])
            )
        for source_shard, indices in indices_by_shard.items():
            collapsed: list[tuple[int, int]] = []
            ordered = sorted(set(indices))
            start = previous = ordered[0]
            for index in ordered[1:]:
                if index == previous + 1:
                    previous = index
                    continue
                collapsed.append(
                    (start * microshard_size, (previous + 1) * microshard_size)
                )
                start = previous = index
            collapsed.append(
                (start * microshard_size, (previous + 1) * microshard_size)
            )
            ranges[(str(process), source_shard)] = collapsed
    return ranges


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def cast_float32(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)


def load_cut1_tables(
    feature_root: Path,
    event_features: Sequence[str],
    jet_features: Sequence[str],
    max_shards_per_process: int | None,
    first_stage_scope: str,
    microshard_manifest: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    detector_scope = first_stage_scope.startswith("detector_")
    event_columns = [
        "process_id",
        "sample_id",
        "event_number",
        "event_id",
        "label",
        "paper_weight",
        "pass_cut1",
        "pass_cut2",
        "pass_cut3",
        "pass_cut4",
        *event_features,
    ]
    if detector_scope:
        event_columns.extend(
            ["hj_btag_wp70_parametric", "lj_btag_wp70_parametric"]
        )
    jet_columns = [
        "process_id",
        "sample_id",
        "event_number",
        "event_id",
        "jet_id",
        "jet_role",
        "truth_flavor",
        "label",
        *jet_features,
    ]
    if detector_scope:
        jet_columns.append("truth_match_valid")
    event_parts: list[pd.DataFrame] = []
    jet_parts: list[pd.DataFrame] = []
    shard_rows: list[dict[str, object]] = []
    selected_ranges = microshard_read_ranges(microshard_manifest)

    for shard in discover_shards(feature_root, max_shards_per_process):
        manifest_path = shard / "feature_manifest.json"
        event_path = shard / "events.parquet"
        jet_path = shard / "jets.parquet"
        if not manifest_path.exists() or not event_path.exists() or not jet_path.exists():
            raise FileNotFoundError(f"Incomplete production feature shard: {shard}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        process = str(manifest["process_id"])
        ranges = selected_ranges.get((process, shard.name))
        if microshard_manifest is not None and not ranges:
            continue
        event_filters: object = [("pass_cut1", "==", True)]
        jet_filters: object = None
        if ranges:
            event_filters = [
                [
                    ("pass_cut1", "==", True),
                    ("event_number", ">=", lower),
                    ("event_number", "<", upper),
                ]
                for lower, upper in ranges
            ]
            jet_filters = [
                [
                    ("event_number", ">=", lower),
                    ("event_number", "<", upper),
                ]
                for lower, upper in ranges
            ]

        events = pd.read_parquet(
            event_path,
            columns=event_columns,
            filters=event_filters,
        )
        require_columns(events, event_columns, str(event_path))
        event_ids = set(events["event_id"].astype(str))

        jets = pd.read_parquet(
            jet_path,
            columns=jet_columns,
            filters=jet_filters,
        )
        require_columns(jets, jet_columns, str(jet_path))
        jets["event_id"] = jets["event_id"].astype(str)
        jets = jets.loc[
            jets["event_id"].isin(event_ids) & jets["jet_role"].isin(ROLE_NAMES)
        ].copy()

        events["event_id"] = events["event_id"].astype(str)
        events["_source_shard"] = shard.name
        cast_float32(events, event_features)
        cast_float32(jets, jet_features)
        events["label"] = events["label"].astype(np.int8)
        jets["label"] = jets["label"].astype(np.int8)
        if not detector_scope:
            jets["truth_match_valid"] = True
        events["paper_weight"] = pd.to_numeric(events["paper_weight"], errors="raise")

        event_parts.append(events)
        jet_parts.append(jets)
        shard_rows.append(
            {
                "shard": str(shard),
                "process": process,
                "manifest_events": int(manifest["n_events"]),
                "cut1_events": int(len(events)),
                "cut1_role_jets": int(len(jets)),
                "microshard_read_ranges": ranges,
            }
        )

    events = pd.concat(event_parts, ignore_index=True)
    jets = pd.concat(jet_parts, ignore_index=True)
    events = events.rename(
        columns={"process_id": "process", "label": "is_signal", "paper_weight": "physics_weight"}
    )
    jets = jets.rename(columns={"process_id": "process", "label": "is_signal"})

    audit = validate_loaded_tables(events, jets, event_features, jet_features)
    audit["shards"] = shard_rows
    return events, jets, audit


def apply_microshard_manifest(
    events: pd.DataFrame,
    jets: pd.DataFrame,
    manifest_path: Path,
    event_features: Sequence[str],
    jet_features: Sequence[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, float] | None,
    float,
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tc_bdt_generated_microshard_split_v1":
        raise ValueError(f"Unsupported microshard manifest: {manifest_path}")
    microshard_size = int(manifest["microshard_size"])
    if microshard_size <= 0:
        raise ValueError("microshard_size must be positive")
    split_method = str(manifest["split_method"])
    if split_method not in {"global_event_hash", "shard_wise_quota"}:
        raise ValueError(f"Unsupported microshard split method: {split_method}")

    selected_units: dict[str, str | None] = {}
    for process, process_payload in dict(manifest["processes"]).items():
        for unit in dict(process_payload)["selected_units"]:
            unit_payload = dict(unit)
            key = (
                f"{process}/{unit_payload['source_shard']}/"
                f"micro_{int(unit_payload['microshard_index']):04d}"
            )
            if key in selected_units:
                raise ValueError(f"Duplicate selected microshard: {key}")
            split = unit_payload.get("split")
            selected_units[key] = None if split is None else str(split)

    source_numbers = pd.to_numeric(events["event_number"], errors="raise").astype(int)
    if (source_numbers < 0).any():
        raise ValueError("Negative source event number")
    microshard_indices = source_numbers // microshard_size
    event_keys = (
        events["process"].astype(str)
        + "/"
        + events["_source_shard"].astype(str)
        + "/micro_"
        + microshard_indices.map(lambda value: f"{value:04d}")
    )
    selected_mask = event_keys.isin(selected_units)
    events = events.loc[selected_mask].copy()
    events["_microshard_index"] = microshard_indices.loc[selected_mask].to_numpy()
    events["_microshard_key"] = event_keys.loc[selected_mask].to_numpy()
    selected_event_ids = set(events["event_id"])
    jets = jets.loc[jets["event_id"].isin(selected_event_ids)].copy()
    if events.empty:
        raise ValueError("Microshard selection produced no detector events")
    filtered_audit = validate_loaded_tables(
        events,
        jets,
        event_features,
        jet_features,
    )

    split_weight_scales = {
        str(split): float(scale)
        for split, scale in dict(manifest["split_weight_scales"]).items()
    }
    global_weight_scale = float(manifest["global_physics_weight_scale"])
    if split_method == "shard_wise_quota":
        events["split"] = events["_microshard_key"].map(selected_units)
        if events["split"].isna().any():
            raise AssertionError("Selected shard-wise events are missing a split")
    elif any(split is not None for split in selected_units.values()):
        raise ValueError("Global event-hash manifest unexpectedly preassigns splits")

    observed_units = set(events["_microshard_key"])
    empty_units = sorted(set(selected_units) - observed_units)
    audit = {
        "manifest": str(manifest_path.resolve()),
        "split_method": split_method,
        "microshard_size": microshard_size,
        "requested_generated_fraction": float(manifest["generated_fraction"]),
        "selected_generated_events": int(manifest["selected_generated_events"]),
        "selected_microshards": len(selected_units),
        "selected_detector_events": int(len(events)),
        "selected_detector_jets": int(len(jets)),
        "empty_after_structural_preselection_microshards": empty_units,
        "filtered_table_audit": filtered_audit,
    }
    return events, jets, audit, split_weight_scales, global_weight_scale


def validate_loaded_tables(
    events: pd.DataFrame,
    jets: pd.DataFrame,
    event_features: Sequence[str],
    jet_features: Sequence[str],
) -> dict[str, object]:
    require_columns(
        events,
        {"event_id", "process", "is_signal", "physics_weight", "pass_cut1", *event_features},
        "assembled Cut1 event table",
    )
    require_columns(
        jets,
        {"event_id", "process", "is_signal", "jet_role", "truth_flavor", *jet_features},
        "assembled Cut1 jet table",
    )
    if events.empty or jets.empty:
        raise ValueError("The assembled Cut1 sample is empty")
    if events["event_id"].duplicated().any():
        examples = events.loc[events["event_id"].duplicated(), "event_id"].head().tolist()
        raise ValueError(f"Duplicate event IDs: {examples}")
    if jets.duplicated(["event_id", "jet_role"]).any():
        examples = jets.loc[
            jets.duplicated(["event_id", "jet_role"]), ["event_id", "jet_role"]
        ].head()
        raise ValueError(f"Duplicate HJ/LJ rows:\n{examples}")
    if not events["pass_cut1"].all():
        raise AssertionError("Non-Cut1 event survived the parquet filter")
    if (events["physics_weight"] <= 0).any():
        raise ValueError("Non-positive paper_weight in Cut1 events")
    if not set(events["is_signal"].unique()).issubset({0, 1}):
        raise ValueError("Event labels are not binary")
    valid_supervision = jets["truth_match_valid"].astype(bool)
    if not set(jets.loc[valid_supervision, "truth_flavor"].astype(str).unique()).issubset(FLAVOR_TO_ID):
        raise ValueError("Unsupported truth_flavor in jet table")

    event_ids = set(events["event_id"])
    jet_event_ids = set(jets["event_id"])
    if event_ids != jet_event_ids:
        raise ValueError(
            f"Event/jet key mismatch: {len(event_ids - jet_event_ids)} without jets, "
            f"{len(jet_event_ids - event_ids)} without events"
        )
    roles = jets.groupby("event_id")["jet_role"].agg(lambda values: tuple(sorted(values)))
    bad_roles = roles[roles != ("HJ", "LJ")]
    if len(bad_roles):
        raise ValueError(f"Events without exactly one HJ and one LJ: {bad_roles.head().to_dict()}")

    consistency = jets.merge(
        events[["event_id", "process", "is_signal"]],
        on="event_id",
        suffixes=("_jet", "_event"),
        validate="many_to_one",
    )
    mismatch = (consistency["process_jet"] != consistency["process_event"]) | (
        consistency["is_signal_jet"] != consistency["is_signal_event"]
    )
    if mismatch.any():
        raise ValueError("Jet process/label metadata disagree with their parent events")

    return {
        "n_cut1_events": int(len(events)),
        "n_role_jets": int(len(jets)),
        "n_processes": int(events["process"].nunique()),
        "process_event_counts": events.groupby("process").size().astype(int).to_dict(),
        "role_counts": jets.groupby("jet_role").size().astype(int).to_dict(),
        "role_flavor_counts": {
            f"{role}:{flavor}": int(count)
            for (role, flavor), count in jets.groupby(["jet_role", "truth_flavor"]).size().items()
        },
        "n_truth_matched_jets": int(valid_supervision.sum()),
        "n_truth_unmatched_jets": int((~valid_supervision).sum()),
        "event_ids_unique": True,
        "exactly_one_hj_lj": True,
        "event_jet_metadata_consistent": True,
    }


def attach_splits_and_weights(
    events: pd.DataFrame,
    jets: pd.DataFrame,
    cfg: core.Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["split"] = core.assign_event_splits(events, cfg)
    return attach_preassigned_splits_and_weights(
        events,
        jets,
        cfg,
        split_weight_scales=None,
    )


def attach_preassigned_splits_and_weights(
    events: pd.DataFrame,
    jets: pd.DataFrame,
    cfg: core.Config,
    split_weight_scales: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    expected_splits = set(SPLITS)
    observed_splits = set(events["split"].astype(str).unique())
    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected preassigned splits {sorted(expected_splits)}, "
            f"found {sorted(observed_splits)}"
        )
    core.assert_disjoint_event_splits(events)
    if split_weight_scales is None:
        events = core.add_split_extrapolation_weights(events)
    else:
        if set(split_weight_scales) != expected_splits:
            raise ValueError(
                "Preassigned split weight scales must define train, validation, and test"
            )
        invalid_scales = {
            split: scale
            for split, scale in split_weight_scales.items()
            if not math.isfinite(scale) or scale <= 0.0
        }
        if invalid_scales:
            raise ValueError(f"Invalid preassigned split weight scales: {invalid_scales}")
        events["physics_weight"] *= events["split"].map(split_weight_scales)
        events["eval_weight"] = events["physics_weight"]
    events["oof_fold"] = np.int8(-1)
    for process, process_events in events.loc[events["split"] == "train"].groupby(
        "process", sort=True
    ):
        folds = [
            core.stable_event_bucket(
                str(process),
                str(event_id),
                cfg.split_seed,
                "tc_bdt_oof_fold_v1",
                cfg.n_folds,
            )
            for event_id in process_events["event_id"]
        ]
        events.loc[process_events.index, "oof_fold"] = np.asarray(
            folds, dtype=np.int8
        )
    if (events.loc[events["split"] == "train", "oof_fold"] < 0).any():
        raise AssertionError("Some training events are missing an OOF fold")
    jets = jets.merge(
        events[["event_id", "split", "oof_fold", "eval_weight"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    if jets[["split", "oof_fold", "eval_weight"]].isna().any().any():
        raise AssertionError("Missing split metadata after event-to-jet merge")
    if int(jets.groupby("event_id")["oof_fold"].nunique().max()) != 1:
        raise AssertionError("The HJ and LJ from one event were assigned different OOF folds")
    return events, jets


def numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    matrix = frame.loc[:, features].to_numpy(dtype=np.float32, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def multiclass_metrics(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, object]:
    result = core.first_stage_metrics(rows["flavor_id"].to_numpy(dtype=int), probabilities)
    try:
        result["macro_ovr_auc_physics_weighted"] = float(
            roc_auc_score(
                rows["flavor_id"].to_numpy(dtype=int),
                probabilities,
                labels=np.arange(3),
                multi_class="ovr",
                average="macro",
                sample_weight=rows["eval_weight"].to_numpy(dtype=float),
            )
        )
    except ValueError:
        result["macro_ovr_auc_physics_weighted"] = math.nan
    return result


def model_device(model: XGBClassifier) -> str:
    config = json.loads(model.get_booster().save_config())
    return str(config["learner"]["generic_param"]["device"])


@contextmanager
def gpu_training_slot(cfg: core.Config, label: str):
    lock_path = os.environ.get("TC_BDT_GPU_LOCK_PATH")
    if cfg.device != "cuda" or not lock_path:
        yield
        return
    slots = int(os.environ.get("TC_BDT_GPU_SLOTS", "1"))
    if slots < 1:
        raise ValueError("TC_BDT_GPU_SLOTS must be positive")
    base_path = Path(lock_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[gpu-lock] waiting: {label}", flush=True)
    handle = None
    slot = -1
    while handle is None:
        for candidate in range(slots):
            candidate_path = Path(f"{base_path}.slot_{candidate}")
            candidate_handle = candidate_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(
                    candidate_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                candidate_handle.close()
                continue
            handle = candidate_handle
            slot = candidate
            break
        if handle is None:
            time.sleep(0.05)
    waited = time.monotonic() - started
    print(
        f"[gpu-lock] acquired slot {slot}/{slots} after {waited:.2f}s: {label}",
        flush=True,
    )
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        print(f"[gpu-lock] released slot {slot}/{slots}: {label}", flush=True)


def fit_shared_flavor_head(
    jets: pd.DataFrame,
    features: Sequence[str],
    cfg: core.Config,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows = jets.copy()
    rows["supervision_valid"] = rows["truth_match_valid"].astype(bool)
    rows["flavor_id"] = rows["truth_flavor"].map(FLAVOR_TO_ID).fillna(-1).astype(np.int8)
    parts = {name: rows.loc[rows["split"] == name].copy() for name in SPLITS}
    matrices = {name: numeric_matrix(part, features) for name, part in parts.items()}
    train = parts["train"]
    oof = np.full((len(train), 3), np.nan, dtype=np.float32)
    ensemble = {
        "validation": np.zeros((len(parts["validation"]), 3), dtype=np.float64),
        "test": np.zeros((len(parts["test"]), 3), dtype=np.float64),
    }
    models: list[XGBClassifier] = []
    weight_audit: list[pd.DataFrame] = []

    for fold in range(cfg.n_folds):
        hold_index = np.flatnonzero(train["oof_fold"].to_numpy(dtype=int) == fold)
        fit_index = np.flatnonzero(
            (train["oof_fold"].to_numpy(dtype=int) != fold)
            & train["supervision_valid"].to_numpy(dtype=bool)
        )
        if not len(hold_index) or not len(fit_index):
            raise AssertionError(f"Shared first-stage fold {fold} is empty")
        fit_rows = train.iloc[fit_index]
        if set(fit_rows["truth_flavor"]) != set(FLAVOR_TO_ID):
            raise AssertionError(f"Shared first-stage fit fold {fold} is missing a flavor class")
        weights = core.flavor_process_balanced_weights(fit_rows)
        audit = fit_rows[["process", "truth_flavor"]].copy()
        audit["training_weight"] = weights
        audit = audit.groupby(["process", "truth_flavor"], as_index=False).agg(
            raw_events=("training_weight", "size"),
            training_weight_sum=("training_weight", "sum"),
        )
        audit["stage"] = "shared_multiclass"
        audit["fold"] = fold
        weight_audit.append(audit)

        with gpu_training_slot(cfg, f"shared_flavor_fold_{fold}"):
            model = core.make_jet_model(cfg, cfg.seed + fold)
            model.fit(
                matrices["train"][fit_index],
                fit_rows["flavor_id"].to_numpy(dtype=int),
                sample_weight=weights,
            )
            oof[hold_index] = model.predict_proba(matrices["train"][hold_index])
            for split in ("validation", "test"):
                ensemble[split] += model.predict_proba(matrices[split])
            model.save_model(model_dir / f"shared_flavor_fold_{fold}.json")
        models.append(model)

    if np.isnan(oof).any():
        raise AssertionError("Shared first-stage OOF predictions contain NaNs")
    for split in ensemble:
        ensemble[split] /= len(models)

    scored_parts: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "features": list(features),
        "n_features": len(features),
        "score_semantics": "Balanced-prior flavor-like scores, not calibrated physical probabilities.",
        "booster_devices": [model_device(model) for model in models],
    }
    for split, predictions in (
        ("train", oof),
        ("validation", ensemble["validation"]),
        ("test", ensemble["test"]),
    ):
        part = parts[split]
        scored = part[["event_id", "jet_role"]].copy()
        scored["p_b"] = predictions[:, FLAVOR_TO_ID["b"]]
        scored["p_c"] = predictions[:, FLAVOR_TO_ID["c"]]
        scored["p_light"] = predictions[:, FLAVOR_TO_ID["light"]]
        scored_parts.append(scored)
        metric_mask = part["supervision_valid"].to_numpy(dtype=bool)
        metrics[split] = multiclass_metrics(part.loc[metric_mask], predictions[metric_mask])
        metrics[split]["n_scored_jets"] = int(len(part))
        metrics[split]["n_supervised_jets"] = int(metric_mask.sum())

    importance_values = np.vstack([model.feature_importances_ for model in models])
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance_values.mean(axis=0),
            "importance_std": importance_values.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    return (
        pd.concat(scored_parts, ignore_index=True),
        metrics,
        importance,
        pd.concat(weight_audit, ignore_index=True),
    )


def binary_process_balanced_weights(rows: pd.DataFrame) -> np.ndarray:
    cell_size = rows.groupby(["binary_target", "process"])["event_id"].transform("size")
    weights = 1.0 / cell_size.to_numpy(dtype=float)
    targets = rows["binary_target"].to_numpy(dtype=int)
    for target in (0, 1):
        mask = targets == target
        weights[mask] /= weights[mask].sum()
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def make_role_model(cfg: core.Config, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
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
        eval_metric="logloss",
        random_state=seed,
        n_jobs=cfg.n_jobs,
        verbosity=0,
    )


def fit_role_head(
    jets: pd.DataFrame,
    role: str,
    target_flavor: str,
    score_column: str,
    seed_offset: int,
    features: Sequence[str],
    cfg: core.Config,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows = jets.loc[jets["jet_role"] == role].copy()
    rows["supervision_valid"] = rows["truth_match_valid"].astype(bool)
    rows["binary_target"] = (rows["truth_flavor"] == target_flavor).astype(np.int8)
    parts = {name: rows.loc[rows["split"] == name].copy() for name in SPLITS}
    matrices = {name: numeric_matrix(part, features) for name, part in parts.items()}
    train = parts["train"]
    oof = np.full(len(train), np.nan, dtype=np.float32)
    ensemble = {
        "validation": np.zeros(len(parts["validation"]), dtype=np.float64),
        "test": np.zeros(len(parts["test"]), dtype=np.float64),
    }
    models: list[XGBClassifier] = []
    weight_audit: list[pd.DataFrame] = []

    for fold in range(cfg.n_folds):
        hold_index = np.flatnonzero(train["oof_fold"].to_numpy(dtype=int) == fold)
        fit_index = np.flatnonzero(
            (train["oof_fold"].to_numpy(dtype=int) != fold)
            & train["supervision_valid"].to_numpy(dtype=bool)
        )
        if not len(hold_index) or not len(fit_index):
            raise AssertionError(f"{role} role-head fold {fold} is empty")
        fit_rows = train.iloc[fit_index]
        if set(fit_rows["binary_target"]) != {0, 1}:
            raise AssertionError(f"{role} role-head fit fold {fold} is missing a target class")
        weights = binary_process_balanced_weights(fit_rows)
        audit = fit_rows[["process", "binary_target"]].copy()
        audit["training_weight"] = weights
        audit = audit.groupby(["process", "binary_target"], as_index=False).agg(
            raw_events=("training_weight", "size"),
            training_weight_sum=("training_weight", "sum"),
        )
        audit["stage"] = f"role_{role.lower()}_{target_flavor}_vs_rest"
        audit["fold"] = fold
        weight_audit.append(audit)

        lock_label = f"role_{role.lower()}_{target_flavor}_fold_{fold}"
        with gpu_training_slot(cfg, lock_label):
            model = make_role_model(cfg, cfg.seed + seed_offset + fold)
            model.fit(
                matrices["train"][fit_index],
                fit_rows["binary_target"].to_numpy(dtype=int),
                sample_weight=weights,
            )
            oof[hold_index] = model.predict_proba(
                matrices["train"][hold_index]
            )[:, 1]
            for split in ("validation", "test"):
                ensemble[split] += model.predict_proba(matrices[split])[:, 1]
            model.save_model(
                model_dir / f"role_{role.lower()}_{target_flavor}_fold_{fold}.json"
            )
        models.append(model)

    if np.isnan(oof).any():
        raise AssertionError(f"{role} role-head OOF predictions contain NaNs")
    for split in ensemble:
        ensemble[split] /= len(models)

    score_parts: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "role": role,
        "target": f"{target_flavor}_vs_rest",
        "features": list(features),
        "n_features": len(features),
        "booster_devices": [model_device(model) for model in models],
    }
    for split, predictions in (
        ("train", oof),
        ("validation", ensemble["validation"]),
        ("test", ensemble["test"]),
    ):
        part = parts[split]
        metric_mask = part["supervision_valid"].to_numpy(dtype=bool)
        metric_part = part.loc[metric_mask]
        metric_predictions = predictions[metric_mask]
        target = metric_part["binary_target"].to_numpy(dtype=int)
        metrics[split] = {
            "n_jets": int(len(part)),
            "n_supervised_jets": int(metric_mask.sum()),
            "positive_fraction": float(target.mean()),
            "auc_unweighted": float(roc_auc_score(target, metric_predictions)),
            "auc_physics_weighted": float(
                roc_auc_score(
                    target,
                    metric_predictions,
                    sample_weight=metric_part["eval_weight"].to_numpy(dtype=float),
                )
            ),
        }
        scored = part[["event_id"]].copy()
        scored[score_column] = predictions
        score_parts.append(scored)

    importance_values = np.vstack([model.feature_importances_ for model in models])
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance_values.mean(axis=0),
            "importance_std": importance_values.std(axis=0),
        }
    ).sort_values("importance_mean", ascending=False)
    return (
        pd.concat(score_parts, ignore_index=True),
        metrics,
        importance,
        pd.concat(weight_audit, ignore_index=True),
    )


def build_stage2_table(
    events: pd.DataFrame,
    jets: pd.DataFrame,
    role_scores: dict[str, pd.DataFrame],
    event_features: Sequence[str],
    jet_features: Sequence[str],
    forbidden_patterns: Sequence[str],
    shared_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = events.copy()
    for task_name in ROLE_TASKS:
        out = out.merge(
            role_scores[task_name], on="event_id", how="left", validate="one_to_one"
        )

    epsilon = 1.0e-6
    hj_b = out["hj_role_p_b_oof"].clip(epsilon, 1.0 - epsilon)
    lj_c = out["lj_role_p_c_oof"].clip(epsilon, 1.0 - epsilon)
    lj_b = out["lj_role_p_b_oof"].clip(epsilon, 1.0 - epsilon)
    out["lj_role_not_b_oof"] = 1.0 - lj_b
    out["lj_role_c_over_b_log_score_oof"] = np.log(lj_c / lj_b)
    out["tc_role_score_product_oof"] = hj_b * lj_c
    out["tc_role_score_contrast_oof"] = hj_b + lj_c - lj_b

    shared_score_features: list[str] = []
    if shared_scores is not None:
        for role, prefix in (("HJ", "hj"), ("LJ", "lj")):
            shared = shared_scores.loc[shared_scores["jet_role"] == role].drop(columns="jet_role")
            shared = shared.rename(
                columns={
                    "p_b": f"{prefix}_p_b_oof",
                    "p_c": f"{prefix}_p_c_oof",
                    "p_light": f"{prefix}_p_light_oof",
                }
            )
            out = out.merge(shared, on="event_id", how="left", validate="one_to_one")
        shared_score_features = [
            "hj_p_b_oof",
            "hj_p_c_oof",
            "hj_p_light_oof",
            "lj_p_b_oof",
            "lj_p_c_oof",
            "lj_p_light_oof",
        ]

    observable_jet_features = [
        feature for feature in jet_features if not feature.startswith("truth_proxy_")
    ]
    for role, prefix in (("HJ", "hj"), ("LJ", "lj")):
        part = jets.loc[jets["jet_role"] == role, ["event_id", *observable_jet_features]].copy()
        part = part.rename(columns={name: f"{prefix}_jet_{name}" for name in observable_jet_features})
        out = out.merge(part, on="event_id", how="left", validate="one_to_one")

    paired_jet_features = [
        f"{prefix}_jet_{feature}"
        for prefix in ("hj", "lj")
        for feature in observable_jet_features
    ]
    feature_sets = {
        "event_only": list(event_features),
        "direct_full": list(dict.fromkeys([*event_features, *paired_jet_features])),
        "double_stage": list(dict.fromkeys([*event_features, *BASE_ROLE_SCORE_FEATURES])),
        "hybrid": list(
            dict.fromkeys([*event_features, *paired_jet_features, *BASE_ROLE_SCORE_FEATURES])
        ),
        "double_stage_role_complete": list(
            dict.fromkeys([*event_features, *ROLE_COMPLETE_SCORE_FEATURES])
        ),
        "hybrid_role_complete": list(
            dict.fromkeys(
                [*event_features, *paired_jet_features, *ROLE_COMPLETE_SCORE_FEATURES]
            )
        ),
    }
    if shared_scores is not None:
        feature_sets.update(
            {
                "double_stage_shared": list(
                    dict.fromkeys([*event_features, *shared_score_features])
                ),
                "hybrid_shared": list(
                    dict.fromkeys([*event_features, *paired_jet_features, *shared_score_features])
                ),
            }
        )
    for model_name, features in feature_sets.items():
        missing = sorted(set(features) - set(out.columns))
        if missing:
            raise ValueError(f"{model_name} missing stage-2 features: {missing}")
        forbidden = [
            feature
            for feature in features
            if any(pattern.lower() in feature.lower() for pattern in forbidden_patterns)
        ]
        if forbidden:
            raise ValueError(f"{model_name} contains forbidden stage-2 features: {forbidden}")
    score_columns = [
        *ROLE_COMPLETE_SCORE_FEATURES,
        *ROLE_DIAGNOSTIC_SCORE_FEATURES,
        *shared_score_features,
    ]
    if out[score_columns].isna().any().any():
        raise AssertionError("Missing OOF flavor scores in stage-2 event table")
    return out, feature_sets


def stage2_training_weight_audit(events: pd.DataFrame) -> pd.DataFrame:
    train = events.loc[events["split"] == "train", ["process", "is_signal", "physics_weight"]].copy()
    train["training_weight"] = core.class_balanced_physics_weights(train)
    return train.groupby(["process", "is_signal"], as_index=False).agg(
        raw_events=("training_weight", "size"),
        physics_weight_sum=("physics_weight", "sum"),
        training_weight_sum=("training_weight", "sum"),
    )


def evaluate_event_model(
    events: pd.DataFrame,
    model_name: str,
    features: Sequence[str],
    cfg: core.Config,
    output: Path,
    seed_offset: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    score_column = f"score_{model_name}"
    with gpu_training_slot(cfg, f"event_{model_name}"):
        model = core.fit_event_model(events, features, cfg, seed_offset=seed_offset)
        model.save_model(output / "models" / f"event_{model_name}.json")
        for split in SPLITS:
            mask = events["split"] == split
            events.loc[mask, score_column] = model.predict_proba(
                core.clean_numeric_frame(events.loc[mask], features)
            )[:, 1]

    validation = events.loc[events["split"] == "validation"].copy()
    test = events.loc[events["split"] == "test"].copy()
    threshold, scan = core.optimize_threshold(
        validation,
        score_column,
        cfg.reference_coefficient,
        cfg.optimization_target_z,
        cfg.threshold_min_raw_signal,
        cfg.threshold_min_raw_background,
    )
    scan.to_csv(output / f"{model_name}_threshold_scan_validation.csv", index=False)
    process_table = core.process_yields(test, score_column, threshold)
    process_table.to_csv(output / f"{model_name}_test_process_yields.csv", index=False)
    signal, background, ratio, z_value = core.yields_at_threshold(test, score_column, threshold)

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
    metrics = {
        "n_features": len(features),
        "features": list(features),
        "booster_device": model_device(model),
        "validation_auc_weighted": float(
            roc_auc_score(
                validation["is_signal"],
                validation[score_column],
                sample_weight=validation["eval_weight"],
            )
        ),
        "test_auc_weighted": float(
            roc_auc_score(
                test["is_signal"],
                test[score_column],
                sample_weight=test["eval_weight"],
            )
        ),
        "test_auc_unweighted": float(roc_auc_score(test["is_signal"], test[score_column])),
        "validation_optimized_threshold": float(threshold),
        "test": {
            "S": signal,
            "B": background,
            "S_over_B": ratio,
            "Z": z_value,
            "Cmin_Z2": core.coefficient_reach(
                signal, background, cfg.reference_coefficient, 2.0
            ),
            "Cmin_Z3": core.coefficient_reach(
                signal, background, cfg.reference_coefficient, 3.0
            ),
        },
    }
    return metrics, events


def evaluate_manual_baseline(events: pd.DataFrame, cfg: core.Config) -> dict[str, float]:
    test = events.loc[events["split"] == "test"]
    selected = test["pass_cut4"].to_numpy(dtype=bool)
    signal, background, ratio, z_value = core.yields_from_mask(test, selected)
    return {
        "S": signal,
        "B": background,
        "S_over_B": ratio,
        "Z": z_value,
        "Cmin_Z2": core.coefficient_reach(signal, background, cfg.reference_coefficient, 2.0),
        "Cmin_Z3": core.coefficient_reach(signal, background, cfg.reference_coefficient, 3.0),
    }


def evaluate_charm_augmented_manual_baseline(
    events: pd.DataFrame,
    cfg: core.Config,
    output: Path,
) -> dict[str, object] | None:
    required = {
        "pass_cut3",
        "hj_btag_wp70_parametric",
        "lj_btag_wp70_parametric",
        "lj_role_p_c_oof",
    }
    if not required.issubset(events.columns):
        return None

    validation = events.loc[events["split"] == "validation"].copy()
    test = events.loc[events["split"] == "test"].copy()

    def base_mask(frame: pd.DataFrame) -> np.ndarray:
        return (
            frame["pass_cut3"].to_numpy(dtype=bool)
            & frame["hj_btag_wp70_parametric"].to_numpy(dtype=bool)
            & ~frame["lj_btag_wp70_parametric"].to_numpy(dtype=bool)
        )

    validation_base = base_mask(validation)
    validation_scores = validation["lj_role_p_c_oof"].to_numpy(dtype=float)
    thresholds = np.unique(
        np.concatenate(
            (
                np.array([0.5]),
                np.quantile(validation_scores[validation_base], np.linspace(0.0, 1.0, 501)),
            )
        )
    )
    labels = validation["is_signal"].to_numpy(dtype=int)
    total_signal = validation.loc[validation["is_signal"] == 1, "eval_weight"].sum()
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        selected = validation_base & (validation_scores >= threshold)
        signal, background, ratio, z_value = core.yields_from_mask(validation, selected)
        rows.append(
            {
                "threshold": float(threshold),
                "S": signal,
                "B": background,
                "S_over_B": ratio,
                "Z": z_value,
                "signal_efficiency": float(signal / total_signal) if total_signal > 0 else 0.0,
                "raw_signal_events": int(np.count_nonzero(selected & (labels == 1))),
                "raw_background_events": int(np.count_nonzero(selected & (labels == 0))),
                "Cmin_Z2": core.coefficient_reach(
                    signal, background, cfg.reference_coefficient, 2.0
                ),
            }
        )
    scan = pd.DataFrame(rows)
    scan.to_csv(output / "manual_cut4_lj_charm_threshold_scan_validation.csv", index=False)
    eligible = scan.loc[
        (scan["signal_efficiency"] >= 0.05)
        & (scan["raw_signal_events"] >= cfg.threshold_min_raw_signal)
        & (scan["raw_background_events"] >= cfg.threshold_min_raw_background)
    ]
    if eligible.empty:
        return {
            "status": "unavailable",
            "reason": "No LJ charm threshold passes the manual-cut stability requirements",
            "validation_max_raw_signal": int(scan["raw_signal_events"].max()),
            "validation_max_raw_background": int(scan["raw_background_events"].max()),
            "validation_max_signal_efficiency": float(scan["signal_efficiency"].max()),
        }
    optimized_threshold = float(
        eligible.sort_values(["Cmin_Z2", "Z", "S_over_B"], ascending=[True, False, False])
        .iloc[0]["threshold"]
    )

    def evaluate(threshold: float) -> dict[str, float]:
        selected = base_mask(test) & (
            test["lj_role_p_c_oof"].to_numpy(dtype=float) >= threshold
        )
        signal, background, ratio, z_value = core.yields_from_mask(test, selected)
        return {
            "threshold": threshold,
            "S": signal,
            "B": background,
            "S_over_B": ratio,
            "Z": z_value,
            "Cmin_Z2": core.coefficient_reach(
                signal, background, cfg.reference_coefficient, 2.0
            ),
            "Cmin_Z3": core.coefficient_reach(
                signal, background, cfg.reference_coefficient, 3.0
            ),
            "raw_selected": int(np.count_nonzero(selected)),
        }

    return {
        "definition": (
            "Cut3 AND HJ b-tagged AND LJ not b-tagged AND LJ role-specific "
            "detector charm score above threshold"
        ),
        "b_tag_source": "truth-matched parametric WP70 flags",
        "c_tag_source": "role-specific detector-feature OOF LJ c-vs-rest score",
        "fixed_threshold_0p5_test": evaluate(0.5),
        "validation_optimized_test": evaluate(optimized_threshold),
    }


def poisson_bootstrap_z(
    test: pd.DataFrame,
    score_column: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if replicates <= 0:
        return {}
    selected = test[score_column].to_numpy(dtype=float) >= threshold
    labels = test["is_signal"].to_numpy(dtype=int)
    weights = test["eval_weight"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    z_values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        multiplicity = rng.poisson(1.0, size=len(test))
        weighted = weights * multiplicity * selected
        signal = float(weighted[labels == 1].sum())
        background = float(weighted[labels == 0].sum())
        z_values[index] = core.significance(signal, background)
    q16, median, q84 = np.quantile(z_values, [0.16, 0.50, 0.84])
    return {
        "replicates": int(replicates),
        "median": float(median),
        "q16": float(q16),
        "q84": float(q84),
        "std": float(z_values.std(ddof=1)),
    }


def write_plots(
    events: pd.DataFrame,
    results: dict[str, object],
    output: Path,
    model_names: Sequence[str],
) -> None:
    test = events.loc[events["split"] == "test"]
    plt.figure(figsize=(7.0, 5.5))
    for model_name in model_names:
        score_column = f"score_{model_name}"
        fpr, tpr, _ = roc_curve(
            test["is_signal"],
            test[score_column],
            sample_weight=test["eval_weight"],
        )
        auc = results["models"][model_name]["test_auc_weighted"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Weighted background efficiency")
    plt.ylabel("Weighted signal efficiency")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "roc_weighted_test.png", dpi=180)
    plt.close()

    comparison_rows = []
    manual = results["manual_cuts"]
    comparison_rows.append(("manual_cuts", manual["Z"], manual["Cmin_Z2"]))
    for model_name in model_names:
        test_metrics = results["models"][model_name]["test"]
        comparison_rows.append((model_name, test_metrics["Z"], test_metrics["Cmin_Z2"]))
    labels = [row[0] for row in comparison_rows]
    z_values = [row[1] for row in comparison_rows]
    c_values = [row[2] for row in comparison_rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(labels, z_values)
    axes[0].set_ylabel("Test Z")
    axes[1].bar(labels, c_values)
    axes[1].set_ylabel(r"$C_{\min}$ at $Z=2$")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output / "model_comparison.png", dpi=180)
    plt.close(figure)


def split_hash(events: pd.DataFrame) -> str:
    values = events[["event_id", "split"]].sort_values("event_id").astype(str)
    payload = "\n".join(values["event_id"] + ":" + values["split"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_stage1_bundle(
    output: Path,
    events: pd.DataFrame,
    role_scores: dict[str, pd.DataFrame],
    role_metrics: dict[str, object],
    shared_scores: pd.DataFrame | None,
    shared_metrics: dict[str, object] | None,
    cfg: core.Config,
    event_features: Sequence[str],
    jet_features: Sequence[str],
    first_stage_scope: str,
) -> None:
    score_table = events[["event_id"]].copy()
    for task_name in ROLE_TASKS:
        score_table = score_table.merge(
            role_scores[task_name],
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    if score_table.isna().any().any():
        raise AssertionError("Stage-1 bundle contains missing role scores")
    score_table.to_parquet(output / "stage1_role_scores.parquet", index=False)
    if shared_scores is not None:
        shared_scores.to_parquet(
            output / "stage1_shared_scores.parquet",
            index=False,
        )
    bundle = {
        "schema_version": STAGE1_BUNDLE_SCHEMA_VERSION,
        "split_sha256": split_hash(events),
        "seed": int(cfg.seed),
        "split_seed": int(cfg.split_seed),
        "folds": int(cfg.n_folds),
        "event_features": list(event_features),
        "jet_features": list(jet_features),
        "first_stage_scope": first_stage_scope,
        "role_specific_stages": role_metrics,
        "shared_flavor_ablation": shared_metrics,
    }
    with (output / "stage1_bundle.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, sort_keys=True, allow_nan=True, default=str)


def load_stage1_bundle(
    stage1_input: Path,
    events: pd.DataFrame,
    cfg: core.Config,
    event_features: Sequence[str],
    jet_features: Sequence[str],
    first_stage_scope: str,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, object],
    pd.DataFrame | None,
    dict[str, object] | None,
]:
    bundle_path = stage1_input / "stage1_bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": STAGE1_BUNDLE_SCHEMA_VERSION,
        "split_sha256": split_hash(events),
        "seed": int(cfg.seed),
        "split_seed": int(cfg.split_seed),
        "folds": int(cfg.n_folds),
        "event_features": list(event_features),
        "jet_features": list(jet_features),
        "first_stage_scope": first_stage_scope,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Incompatible stage-1 bundle: {mismatches}")
    score_table = pd.read_parquet(stage1_input / "stage1_role_scores.parquet")
    if score_table["event_id"].duplicated().any():
        raise ValueError("Stage-1 role scores contain duplicate event IDs")
    if set(score_table["event_id"]) != set(events["event_id"]):
        raise ValueError("Stage-1 role scores do not match current events")
    role_scores = {
        task_name: score_table[["event_id", str(task["column"])]].copy()
        for task_name, task in ROLE_TASKS.items()
    }
    shared_path = stage1_input / "stage1_shared_scores.parquet"
    shared_scores = pd.read_parquet(shared_path) if shared_path.is_file() else None
    return (
        role_scores,
        dict(payload["role_specific_stages"]),
        shared_scores,
        payload.get("shared_flavor_ablation"),
    )


def merge_parallel_model_outputs(
    stage1_dir: Path,
    model_outputs: dict[str, Path],
    output: Path,
) -> None:
    """Merge independently trained stage-2 jobs into the legacy cell layout."""
    if not model_outputs:
        raise ValueError("No model outputs supplied for merge")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}
    score_frames: dict[str, pd.DataFrame] = {}
    feature_sets: dict[str, list[str]] = {}
    for model_name, model_output in model_outputs.items():
        summary = json.loads(
            (model_output / "summary.json").read_text(encoding="utf-8")
        )
        if set(summary["models"]) != {model_name}:
            raise ValueError(
                f"Model job {model_output} does not contain only {model_name}"
            )
        summaries[model_name] = summary
        scores = pd.read_parquet(model_output / "event_scores_and_splits.parquet")
        score_frames[model_name] = scores
        manifest = json.loads(
            (model_output / "feature_manifest.json").read_text(encoding="utf-8")
        )
        feature_sets[model_name] = list(
            manifest["stage2_feature_sets"][model_name]
        )

    model_names = tuple(model_outputs)
    first = summaries[model_names[0]]
    invariant_keys = ("config", "input_audit", "role_specific_stages", "manual_cuts")
    for model_name in model_names[1:]:
        current = summaries[model_name]
        for key in invariant_keys:
            if current[key] != first[key]:
                raise RuntimeError(
                    f"Parallel model output mismatch for {key}: {model_name}"
                )

    merged = dict(first)
    merged["models"] = {
        model_name: summaries[model_name]["models"][model_name]
        for model_name in model_names
    }
    pipeline_options = dict(first["pipeline_options"])
    pipeline_options["selected_stage2_models"] = list(model_names)
    pipeline_options["stage2_seed_offsets"] = {
        model_name: STAGE2_SEED_OFFSETS[model_name]
        for model_name in model_names
    }
    pipeline_options["parallel_model_execution"] = {
        "enabled": True,
        "shared_stage1_dir": str(stage1_dir.resolve()),
        "model_outputs": {
            name: str(path.resolve()) for name, path in model_outputs.items()
        },
    }
    merged["pipeline_options"] = pipeline_options

    first_scores = score_frames[model_names[0]]
    common_columns = [
        column for column in first_scores.columns if not column.startswith("score_")
    ]
    combined_scores = first_scores[common_columns].copy()
    for model_name in model_names:
        frame = score_frames[model_name][["event_id", f"score_{model_name}"]]
        combined_scores = combined_scores.merge(
            frame,
            on="event_id",
            how="left",
            validate="one_to_one",
        )
    combined_scores.to_parquet(
        output / "event_scores_and_splits.parquet",
        index=False,
    )

    feature_manifest = json.loads(
        (next(iter(model_outputs.values())) / "feature_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    feature_manifest["stage2_feature_sets"] = feature_sets
    (output / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copy2(stage1_dir / "input_audit.json", output / "input_audit.json")
    shutil.copy2(
        stage1_dir / "first_stage_training_weight_audit.csv",
        output / "first_stage_training_weight_audit.csv",
    )
    shutil.copy2(
        stage1_dir / "stage1_bundle.json",
        output / "shared_stage1_bundle.json",
    )
    for artifact in stage1_dir.glob("role_*_feature_importance.csv"):
        shutil.copy2(artifact, output / artifact.name)
    if (stage1_dir / "shared_flavor_feature_importance.csv").is_file():
        shutil.copy2(
            stage1_dir / "shared_flavor_feature_importance.csv",
            output / "shared_flavor_feature_importance.csv",
        )
    for artifact in (stage1_dir / "models").glob("*.json"):
        shutil.copy2(artifact, output / "models" / artifact.name)
    for model_name, model_output in model_outputs.items():
        for artifact in (model_output / "models").glob(f"*{model_name}*.json"):
            shutil.copy2(artifact, output / "models" / artifact.name)
        importance = model_output / f"{model_name}_feature_importance.csv"
        if importance.is_file():
            shutil.copy2(importance, output / importance.name)

    auc_rows = []
    for model_name, metrics in merged["models"].items():
        auc_rows.append(
            {
                "model": model_name,
                "validation_auc_weighted": metrics["validation_auc_weighted"],
                "test_auc_weighted": metrics["test_auc_weighted"],
                "test_auc_unweighted": metrics["test_auc_unweighted"],
                **metrics["test"],
            }
        )
    pd.DataFrame(auc_rows).to_csv(output / "model_comparison.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True, allow_nan=True, default=str),
        encoding="utf-8",
    )
    write_plots(combined_scores, merged, output, model_names)


def main() -> int:
    args = parse_args()
    if args.stage1_only and args.stage1_input is not None:
        raise ValueError("--stage1-only and --stage1-input are mutually exclusive")
    if args.stage1_only and args.models:
        raise ValueError("--stage1-only does not accept --models")
    if not math.isfinite(args.physics_weight_scale) or args.physics_weight_scale <= 0.0:
        raise ValueError("--physics-weight-scale must be finite and positive")
    if args.microshard_manifest is not None and not math.isclose(
        args.physics_weight_scale,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "--physics-weight-scale must remain 1 with --microshard-manifest"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models").mkdir(parents=True, exist_ok=True)
    event_features, jet_features, forbidden_patterns, first_stage_scope = read_contract(
        args.feature_contract
    )

    cfg = core.Config(
        seed=args.seed,
        split_seed=args.split_seed,
        n_folds=args.folds,
        n_jobs=args.n_jobs,
        device=args.device,
        bootstrap_replicates=args.bootstrap,
        reference_coefficient=args.reference_coefficient,
        threshold_min_raw_signal=args.threshold_min_raw_signal,
        threshold_min_raw_background=args.threshold_min_raw_background,
        jet_n_estimators=args.jet_n_estimators,
        jet_max_depth=args.jet_max_depth,
        jet_learning_rate=args.jet_learning_rate,
        event_n_estimators=args.event_n_estimators,
        event_max_depth=args.event_max_depth,
        event_learning_rate=args.event_learning_rate,
    )

    print("[1/6] Loading and auditing Cut1 production tables", flush=True)
    feature_root = args.feature_root
    if args.microshard_manifest is not None:
        microshard_payload = json.loads(
            args.microshard_manifest.read_text(encoding="utf-8")
        )
        feature_root = Path(str(microshard_payload["feature_index"]))
    events, jets, input_audit = load_cut1_tables(
        feature_root,
        event_features,
        jet_features,
        args.max_shards_per_process,
        first_stage_scope,
        microshard_payload if args.microshard_manifest is not None else None,
    )
    split_weight_scales: dict[str, float] | None = None
    effective_global_weight_scale = float(args.physics_weight_scale)
    if args.microshard_manifest is not None:
        (
            events,
            jets,
            microshard_audit,
            split_weight_scales,
            effective_global_weight_scale,
        ) = apply_microshard_manifest(
            events,
            jets,
            args.microshard_manifest,
            event_features,
            jet_features,
        )
        input_audit["microshard_selection"] = microshard_audit
    if args.mjj_min_gev > 0.0:
        require_columns(events, ["mjj"], "Mjj preselection")
        before_events = len(events)
        events = events.loc[events["mjj"] >= args.mjj_min_gev].copy()
        selected_event_ids = set(events["event_id"])
        jets = jets.loc[jets["event_id"].isin(selected_event_ids)].copy()
        filtered_audit = validate_loaded_tables(events, jets, event_features, jet_features)
        input_audit["additional_mjj_preselection"] = {
            "mjj_min_gev": float(args.mjj_min_gev),
            "events_before": int(before_events),
            "events_after": int(len(events)),
            "signal_events_after": int(events["is_signal"].sum()),
            "process_event_counts_after": filtered_audit["process_event_counts"],
        }
    weight_sum_before_scale = float(events["physics_weight"].sum())
    split_method = (
        None
        if args.microshard_manifest is None
        else str(microshard_audit["split_method"])
    )
    if split_weight_scales is None:
        events["physics_weight"] *= effective_global_weight_scale
        events, jets = attach_splits_and_weights(events, jets, cfg)
        applied_scale: object = effective_global_weight_scale
    else:
        if split_method == "global_event_hash":
            events["split"] = core.assign_event_splits(events, cfg)
        events, jets = attach_preassigned_splits_and_weights(
            events,
            jets,
            cfg,
            split_weight_scales,
        )
        applied_scale = split_weight_scales
    input_audit["physics_weight_scaling"] = {
        "scale": applied_scale,
        "sum_before": weight_sum_before_scale,
        "sum_after": float(events["physics_weight"].sum()),
        "process_sums_after": {
            str(process): float(weight)
            for process, weight in events.groupby("process")["physics_weight"].sum().items()
        },
    }
    input_audit["split_event_counts"] = {
        f"{split}:{process}": int(count)
        for (split, process), count in events.groupby(["split", "process"]).size().items()
    }
    input_audit["oof_fold_process_counts"] = {
        f"{fold}:{process}": int(count)
        for (fold, process), count in events.loc[events["split"] == "train"]
        .groupby(["oof_fold", "process"])
        .size()
        .items()
    }
    input_audit["split_sha256"] = split_hash(events)
    with (args.output / "input_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(input_audit, handle, indent=2, sort_keys=True, default=str)

    if args.stage1_input is None:
        print("[2/6] Training role-specific HJ-b, LJ-c, and LJ-b OOF heads", flush=True)
        role_scores: dict[str, pd.DataFrame] = {}
        role_metrics: dict[str, object] = {}
        weight_audits: list[pd.DataFrame] = []
        for task_name, task in ROLE_TASKS.items():
            scores, metrics, importance, weights = fit_role_head(
                jets,
                task["role"],
                task["target"],
                task["column"],
                task["seed_offset"],
                jet_features,
                cfg,
                args.output / "models",
            )
            role_scores[task_name] = scores
            role_metrics[task_name] = metrics
            importance.to_csv(
                args.output
                / f"role_{task['role'].lower()}_{task['target']}_feature_importance.csv",
                index=False,
            )
            weight_audits.append(weights)

        shared_scores: pd.DataFrame | None = None
        shared_metrics: dict[str, object] | None = None
        if args.include_shared_ablation:
            print("[3/6] Training optional shared b/c/light OOF ablation", flush=True)
            shared_scores, shared_metrics, shared_importance, shared_weights = fit_shared_flavor_head(
                jets,
                jet_features,
                cfg,
                args.output / "models",
            )
            shared_importance.to_csv(
                args.output / "shared_flavor_feature_importance.csv", index=False
            )
            weight_audits.append(shared_weights)
        else:
            print("[3/6] Shared b/c/light ablation disabled", flush=True)

        pd.concat(weight_audits, ignore_index=True).to_csv(
            args.output / "first_stage_training_weight_audit.csv", index=False
        )
        if args.stage1_only:
            write_stage1_bundle(
                args.output,
                events,
                role_scores,
                role_metrics,
                shared_scores,
                shared_metrics,
                cfg,
                event_features,
                jet_features,
                first_stage_scope,
            )
            print("[3/6] Shared stage-1 bundle complete", flush=True)
            return 0
    else:
        print(f"[2/6] Reusing shared stage-1 bundle: {args.stage1_input}", flush=True)
        role_scores, role_metrics, shared_scores, shared_metrics = load_stage1_bundle(
            args.stage1_input,
            events,
            cfg,
            event_features,
            jet_features,
            first_stage_scope,
        )
        if args.include_shared_ablation != (shared_scores is not None):
            raise RuntimeError(
                "Shared-ablation setting differs from the stage-1 bundle"
            )
        print("[3/6] Shared OOF scores loaded without retraining", flush=True)

    print("[4/6] Building leakage-guarded stage-2 feature sets", flush=True)
    events, feature_sets = build_stage2_table(
        events,
        jets,
        role_scores,
        event_features,
        jet_features,
        forbidden_patterns,
        shared_scores=shared_scores,
    )
    if args.models:
        unknown_models = sorted(set(args.models) - set(feature_sets))
        if unknown_models:
            raise ValueError(
                f"Unknown --models entries {unknown_models}; available={sorted(feature_sets)}"
            )
        requested_models = list(dict.fromkeys(args.models))
        feature_sets = {name: feature_sets[name] for name in requested_models}
    model_names = tuple(feature_sets)
    stage2_training_weight_audit(events).to_csv(
        args.output / "stage2_training_weight_audit.csv", index=False
    )
    with (args.output / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "event_contract_features": event_features,
                "first_stage_features": jet_features,
                "first_stage_scope": first_stage_scope,
                "first_stage_default": "role_specific_hj_b_lj_c_and_lj_b",
                "role_complete_score_features": ROLE_COMPLETE_SCORE_FEATURES,
                "shared_ablation_enabled": args.include_shared_ablation,
                "stage2_feature_sets": feature_sets,
                "stage2_truth_proxy_policy": "forbidden; only OOF model scores may transfer flavor information",
            },
            handle,
            indent=2,
        )

    print("[5/6] Training CUDA event-level controls and double-stage models", flush=True)
    results: dict[str, object] = {
        "config": asdict(cfg),
        "pipeline_options": {
            "first_stage_default": "role_specific_hj_b_lj_c_and_lj_b",
            "shared_ablation_enabled": args.include_shared_ablation,
            "additional_mjj_preselection_gev": float(args.mjj_min_gev),
            "physics_weight_scale": applied_scale,
            "microshard_manifest": (
                None
                if args.microshard_manifest is None
                else str(args.microshard_manifest.resolve())
            ),
            "selected_stage2_models": list(model_names),
            "stage2_seed_offsets": {
                model_name: STAGE2_SEED_OFFSETS[model_name]
                for model_name in model_names
            },
        },
        "input_audit": input_audit,
        "role_specific_stages": role_metrics,
        "manual_cuts": evaluate_manual_baseline(events, cfg),
        "models": {},
    }
    charm_manual = evaluate_charm_augmented_manual_baseline(events, cfg, args.output)
    if charm_manual is not None:
        results["manual_cut4_hj_b_lj_notb_plus_lj_c"] = charm_manual
    if shared_metrics is not None:
        results["shared_flavor_ablation"] = shared_metrics
    for model_name in model_names:
        print(f"  - {model_name}", flush=True)
        metrics, events = evaluate_event_model(
            events,
            model_name,
            feature_sets[model_name],
            cfg,
            args.output,
            seed_offset=STAGE2_SEED_OFFSETS[model_name],
        )
        metrics["bootstrap_test_Z"] = poisson_bootstrap_z(
            events.loc[events["split"] == "test"],
            f"score_{model_name}",
            metrics["validation_optimized_threshold"],
            cfg.bootstrap_replicates,
            cfg.seed + 10_000 + STAGE2_SEED_OFFSETS[model_name],
        )
        results["models"][model_name] = metrics

    score_columns = [
        "event_id",
        "process",
        "is_signal",
        "physics_weight",
        "eval_weight",
        "split",
        "pass_cut2",
        "pass_cut3",
        "pass_cut4",
        "hj_role_p_b_oof",
        "lj_role_p_c_oof",
        "lj_role_p_b_oof",
        "lj_role_not_b_oof",
        "lj_role_c_over_b_log_score_oof",
        "tc_role_score_product_oof",
        "tc_role_score_contrast_oof",
    ]
    if {"hj_btag_wp70_parametric", "lj_btag_wp70_parametric"}.issubset(events.columns):
        score_columns.extend(
            ["hj_btag_wp70_parametric", "lj_btag_wp70_parametric"]
        )
    if shared_scores is not None:
        score_columns.extend(
            [
                "hj_p_b_oof",
                "hj_p_c_oof",
                "hj_p_light_oof",
                "lj_p_b_oof",
                "lj_p_c_oof",
                "lj_p_light_oof",
            ]
        )
    score_columns.extend(f"score_{name}" for name in model_names)
    events[score_columns].to_parquet(args.output / "event_scores_and_splits.parquet", index=False)

    auc_rows = []
    for model_name, metrics in results["models"].items():
        auc_rows.append(
            {
                "model": model_name,
                "validation_auc_weighted": metrics["validation_auc_weighted"],
                "test_auc_weighted": metrics["test_auc_weighted"],
                "test_auc_unweighted": metrics["test_auc_unweighted"],
                **metrics["test"],
            }
        )
    pd.DataFrame(auc_rows).to_csv(args.output / "model_comparison.csv", index=False)

    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, allow_nan=True, default=str)
    write_plots(events, results, args.output, model_names)
    print("[6/6] Complete", flush=True)
    print(json.dumps({"output": str(args.output), "models": results["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
