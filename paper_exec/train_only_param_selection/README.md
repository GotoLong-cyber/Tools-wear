# Train-Only Retrieval 超参数选择说明

本目录用于执行并记录 `KNN-DRR` 的 **train-only validation** 超参数选择流程。

## 目标

在不访问 outer test fold 的前提下，为正式 retrieval 协议选择一组可冻结的固定参数：

- `k`
- `beta`
- `late_q`

## 协议

对每个 outer fold：

1. 使用已经训练完成的 **clean retrieval-backbone checkpoint**
2. 使用 source-domain `train` split 构建 retrieval library
3. 使用同一 source-domain runs 的 `val` split 作为 validation queries
4. 在 validation 上比较候选参数网格
5. 记录每折最优结果
6. 按三折 validation MAE 平均值选择 **global fixed config**

## 重要边界

- 不使用 outer test fold (`c6 / c1 / c4`) 做参数选择
- 不在 test 结果上挑最好参数
- 正式论文主结果只能使用这里选出的冻结参数
- 如果后续在 test 上做参数敏感性分析，必须标注为 exploratory

## 主要产物

- `results/train_only_knn_grid_all.csv`
- `results/train_only_knn_fold_best.csv`
- `results/train_only_knn_global_summary.csv`
- `results/selected_knn_config_train_only.json`
- `manifests/仅训练选参汇总.md`
