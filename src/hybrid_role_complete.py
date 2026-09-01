"""Minimal two-stage Hybrid Role-Complete BDT implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SPLITS = ("train", "validation", "test")
REQUIRED_EVENT_COLUMNS = {"event_id", "label", "split"}
REQUIRED_JET_COLUMNS = {"event_id", "role", "truth_flavor"}
FORBIDDEN_FEATURE_TOKENS = {
    "event_id",
    "label",
    "split",
    "truth",
    "flavor",
    "flavour",
    "target",
}


@dataclass(frozen=True)
class Stage1Task:
    """Definition of one role-specific binary tagger."""

    name: str
    role: str
    positive_flavor: str


DEFAULT_TASKS = (
    Stage1Task("hj_role_p_b", "HJ", "b"),
    Stage1Task("lj_role_p_c", "LJ", "c"),
    Stage1Task("lj_role_p_b", "LJ", "b"),
)


class HybridRoleCompleteBDT:
    """Train role taggers with OOF scores and one event-level XGBoost model.

    The class operates on already reconstructed features. It deliberately does
    not implement event selection, physical weights, or significance metrics.
    """

    def __init__(
        self,
        event_features: Sequence[str],
        jet_features: Sequence[str],
        *,
        tasks: Sequence[Stage1Task] = DEFAULT_TASKS,
        n_folds: int = 5,
        seed: int = 17,
        device: str = "cpu",
        n_jobs: int = 1,
        stage1_params: dict | None = None,
        stage2_params: dict | None = None,
    ) -> None:
        self.event_features = list(event_features)
        self.jet_features = list(jet_features)
        self.tasks = tuple(tasks)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.device = str(device)
        self.n_jobs = int(n_jobs)
        self.stage1_params = dict(stage1_params or {})
        self.stage2_params = dict(stage2_params or {})

        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not self.event_features or not self.jet_features:
            raise ValueError("event_features and jet_features must be non-empty")
        self._check_feature_names(self.event_features)
        self._check_feature_names(self.jet_features)

    @staticmethod
    def _check_feature_names(features: Sequence[str]) -> None:
        duplicates = sorted({name for name in features if features.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate model features: {duplicates}")
        forbidden = [
            name
            for name in features
            if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"Forbidden supervision/metadata features: {forbidden}")

    def _new_model(self, stage: int, seed_offset: int) -> XGBClassifier:
        defaults = {
            "n_estimators": 500 if stage == 1 else 700,
            "max_depth": 6 if stage == 1 else 5,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": self.device,
            "n_jobs": self.n_jobs,
            "random_state": self.seed + seed_offset,
        }
        defaults.update(self.stage1_params if stage == 1 else self.stage2_params)
        return XGBClassifier(**defaults)

    def _validate_tables(
        self,
        events: pd.DataFrame,
        jets: pd.DataFrame,
        *,
        require_splits: bool,
        require_truth: bool,
    ) -> None:
        event_required = set(REQUIRED_EVENT_COLUMNS)
        if not require_splits:
            event_required.remove("split")
            event_required.remove("label")
        missing_events = sorted(event_required - set(events.columns))
        jet_required = set(REQUIRED_JET_COLUMNS)
        if not require_truth:
            jet_required.remove("truth_flavor")
        missing_jets = sorted(jet_required - set(jets.columns))
        if missing_events or missing_jets:
            raise ValueError(
                f"Missing columns: events={missing_events}, jets={missing_jets}"
            )
        missing_features = sorted(
            (set(self.event_features) - set(events.columns))
            | (set(self.jet_features) - set(jets.columns))
        )
        if missing_features:
            raise ValueError(f"Missing model features: {missing_features}")
        if events["event_id"].duplicated().any():
            raise ValueError("events must contain one row per event_id")
        if require_splits and set(events["split"].astype(str)) != set(SPLITS):
            raise ValueError(f"split must contain exactly {SPLITS}")

        event_ids = set(events["event_id"])
        selected_jets = jets.loc[jets["event_id"].isin(event_ids)]
        counts = selected_jets.groupby(["event_id", "role"]).size().unstack(fill_value=0)
        for role in {task.role for task in self.tasks}:
            if role not in counts or not (counts[role] == 1).all():
                raise ValueError(f"Every event must contain exactly one {role} jet")

    @staticmethod
    def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        matrix = frame.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        return matrix

    def _raw_event_frame(
        self,
        events: pd.DataFrame,
        jets: pd.DataFrame,
    ) -> pd.DataFrame:
        event_index = events.set_index("event_id", verify_integrity=True)
        output = event_index.loc[:, self.event_features].copy()
        for role in sorted({task.role for task in self.tasks}):
            role_rows = jets.loc[
                jets["role"].eq(role) & jets["event_id"].isin(event_index.index)
            ].set_index("event_id", verify_integrity=True)
            renamed = role_rows.loc[:, self.jet_features].rename(
                columns={name: f"{role.lower()}_{name}" for name in self.jet_features}
            )
            output = output.join(renamed, how="left", validate="one_to_one")
        return output

    @staticmethod
    def _add_role_combinations(scores: pd.DataFrame) -> pd.DataFrame:
        output = scores.copy()
        eps = 1.0e-6
        hj_b = output["hj_role_p_b"].clip(eps, 1.0 - eps)
        lj_c = output["lj_role_p_c"].clip(eps, 1.0 - eps)
        lj_b = output["lj_role_p_b"].clip(eps, 1.0 - eps)
        output["lj_role_c_over_b_log_score"] = np.log(lj_c / lj_b)
        output["tc_role_score_product"] = hj_b * lj_c
        output["tc_role_score_contrast"] = hj_b + lj_c - lj_b
        return output

    def _fit_stage1(
        self,
        events: pd.DataFrame,
        jets: pd.DataFrame,
        fold_by_event: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, list[XGBClassifier]]]:
        event_index = events.set_index("event_id", verify_integrity=True)
        scores = pd.DataFrame(index=event_index.index)
        models: dict[str, list[XGBClassifier]] = {}

        for task_index, task in enumerate(self.tasks):
            rows = jets.loc[
                jets["role"].eq(task.role)
                & jets["event_id"].isin(event_index.index)
            ].set_index("event_id", verify_integrity=True).loc[event_index.index]
            labels = rows["truth_flavor"].astype(str).eq(task.positive_flavor).astype(np.int8)
            features = self._matrix(rows, self.jet_features)
            task_scores = np.full(len(rows), np.nan, dtype=np.float32)
            task_models: list[XGBClassifier] = []

            train_mask = event_index["split"].eq("train").to_numpy()
            for fold in range(self.n_folds):
                hold_mask = train_mask & fold_by_event.eq(fold).to_numpy()
                fit_mask = train_mask & ~hold_mask
                if labels.loc[fit_mask].nunique() != 2:
                    raise ValueError(f"Stage-1 task {task.name} lacks both classes")
                model = self._new_model(stage=1, seed_offset=100 * task_index + fold)
                model.fit(features[fit_mask], labels.to_numpy()[fit_mask])
                task_scores[hold_mask] = model.predict_proba(features[hold_mask])[:, 1]
                task_models.append(model)

            evaluation_mask = ~train_mask
            task_scores[evaluation_mask] = np.mean(
                [
                    model.predict_proba(features[evaluation_mask])[:, 1]
                    for model in task_models
                ],
                axis=0,
            )
            if not np.isfinite(task_scores).all():
                raise RuntimeError(f"Incomplete OOF scores for {task.name}")
            scores[task.name] = task_scores
            models[task.name] = task_models

        return self._add_role_combinations(scores), models

    def fit(self, events: pd.DataFrame, jets: pd.DataFrame) -> "HybridRoleCompleteBDT":
        """Fit all Stage-1 folds and the Stage-2 event classifier."""

        events = events.copy()
        jets = jets.copy()
        events["event_id"] = events["event_id"].astype(str)
        jets["event_id"] = jets["event_id"].astype(str)
        self._validate_tables(events, jets, require_splits=True, require_truth=True)

        event_index = events.set_index("event_id", verify_integrity=True)
        train_events = event_index.loc[event_index["split"].eq("train")]
        splitter = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed,
        )
        fold_by_event = pd.Series(-1, index=event_index.index, dtype=np.int16)
        for fold, (_, hold_positions) in enumerate(
            splitter.split(train_events.index, train_events["label"])
        ):
            fold_by_event.loc[train_events.index[hold_positions]] = fold

        role_scores, self.stage1_models_ = self._fit_stage1(
            events, jets, fold_by_event
        )
        raw_features = self._raw_event_frame(events, jets)
        stage2 = raw_features.join(role_scores, how="left", validate="one_to_one")
        self.stage2_feature_names_ = list(stage2.columns)
        labels = event_index["label"].astype(np.int8)
        train_mask = event_index["split"].eq("train")

        self.stage2_model_ = self._new_model(stage=2, seed_offset=10_000)
        self.stage2_model_.fit(
            self._matrix(stage2.loc[train_mask], self.stage2_feature_names_),
            labels.loc[train_mask].to_numpy(),
        )

        all_scores = pd.Series(
            self.stage2_model_.predict_proba(
                self._matrix(stage2, self.stage2_feature_names_)
            )[:, 1],
            index=event_index.index,
            name="score",
        )
        self.validation_scores_ = all_scores.loc[event_index["split"].eq("validation")]
        self.test_scores_ = all_scores.loc[event_index["split"].eq("test")]
        self.metrics_ = {
            split: float(
                roc_auc_score(
                    labels.loc[event_index["split"].eq(split)],
                    all_scores.loc[event_index["split"].eq(split)],
                )
            )
            for split in ("validation", "test")
        }
        return self

    def predict_proba(
        self,
        events: pd.DataFrame,
        jets: pd.DataFrame,
    ) -> pd.Series:
        """Predict signal probabilities for new events with trained fold ensembles."""

        if not hasattr(self, "stage2_model_"):
            raise RuntimeError("Call fit before predict_proba")
        events = events.copy()
        jets = jets.copy()
        events["event_id"] = events["event_id"].astype(str)
        jets["event_id"] = jets["event_id"].astype(str)
        self._validate_tables(events, jets, require_splits=False, require_truth=False)

        event_index = events.set_index("event_id", verify_integrity=True)
        role_scores = pd.DataFrame(index=event_index.index)
        for task in self.tasks:
            rows = jets.loc[
                jets["role"].eq(task.role)
                & jets["event_id"].isin(event_index.index)
            ].set_index("event_id", verify_integrity=True).loc[event_index.index]
            features = self._matrix(rows, self.jet_features)
            role_scores[task.name] = np.mean(
                [model.predict_proba(features)[:, 1] for model in self.stage1_models_[task.name]],
                axis=0,
            )

        stage2 = self._raw_event_frame(events, jets).join(
            self._add_role_combinations(role_scores),
            how="left",
            validate="one_to_one",
        )
        scores = self.stage2_model_.predict_proba(
            self._matrix(stage2, self.stage2_feature_names_)
        )[:, 1]
        return pd.Series(scores, index=event_index.index, name="score")
