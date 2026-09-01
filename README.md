# Hybrid Role-Complete BDT

This repository contains the two-stage XGBoost training core used in the 10 TeV `tc` analysis. Event datasets, trained model weights, and cluster-specific scripts are not included.

## Architecture

1. Stage 1 trains three role-specific taggers: `HJ -> b`, `LJ -> c`, and `LJ -> b`. Training-event predictions are produced with strict five-fold out-of-fold evaluation.
2. Stage 2 combines reconstructed event observables, detector-level HJ/LJ features, and six out-of-fold role-score features in the Hybrid Role-Complete classifier.
3. The validation split selects the score threshold, while the test split is used only for the final AUC and S/B/Z evaluation. Truth flavor is used only as Stage-1 supervision.

## Input layout

```text
FEATURE_ROOT/
  PROCESS/
    shard_0000/
      events.parquet
      jets.parquet
      feature_manifest.json
```

The ordered feature contract and forbidden model inputs are defined in `configs/BDT_FEATURES_DETECTOR.yaml`. Every event must have a unique `event_id` and one `HJ` plus one `LJ` row in the jet table.

## Installation and training

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train_hybrid_role_complete.py \
  --feature-root /path/to/features \
  --output outputs/run_001 \
  --models hybrid_role_complete \
  --device cpu \
  --n-jobs 8 \
  --mjj-min-gev 7000
```

For CUDA training, replace `--device cpu` with `--device cuda`. Main outputs include trained models, `summary.json`, `model_comparison.csv`, and event scores with their frozen split assignments.

## Detector caveat

The Detector V2 impact-parameter features form a parameterized tracking-sensitivity benchmark rather than a validated muon-collider detector-performance claim. Published results should therefore include the no-tracking/IP ablation.
