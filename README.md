# Hybrid Role-Complete BDT

这是 10 TeV `tc` 分析中使用的两层 XGBoost 核心代码，不包含事件数据、模型权重或服务器脚本。

模型结构：

1. 第一层分别训练 `HJ -> b`、`LJ -> c`、`LJ -> b` 三个 tagger；训练事件使用严格 5-fold OOF 分数。
2. 第二层 Hybrid Role-Complete 使用事件级重建量、HJ/LJ detector features，以及六个 OOF role-score 特征完成 signal/background 分类。
3. validation 用于选择阈值，test 只用于最终 AUC、S/B/Z 评估；truth flavor 只作为第一层监督标签。

## 输入

```text
FEATURE_ROOT/
  PROCESS/
    shard_0000/
      events.parquet
      jets.parquet
      feature_manifest.json
```

字段顺序和禁止进入模型的字段由 `configs/BDT_FEATURES_DETECTOR.yaml` 固定。每个事件必须有唯一 `event_id`，每个事件应对应一个 `HJ` 和一个 `LJ` jet row。

## 运行

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

使用 CUDA 时将 `--device cpu` 改为 `--device cuda`。主要输出包括模型文件、`summary.json`、`model_comparison.csv` 和带固定 split 的事件分数。

注意：Detector V2 的 IP 特征属于参数化 tracking sensitivity benchmark；公开结果应同时报告 no-tracking/IP 消融。
