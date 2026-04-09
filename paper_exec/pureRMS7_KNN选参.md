# pureRMS7_KNN选参

## 终版合法选参结论（Baseline + KNN）

以下结果是**当前论文终版应引用的唯一合法选参来源**：

- 主线：`pure RMS7 + Baseline + KNN`
- 训练协议：`wear_agg=mean`，`no validation set`
- 固定轮数：`epoch=300`
- 选参协议：`source-only inner-LOSO`

### 合法选参网格

- `k ∈ {3,5,7,10}`
- `beta ∈ {0.3,0.5,0.7,1.0}`
- `late_q ∈ {0.0,0.5,0.8}`
- 选择指标：`mae_full_raw`

### 合法选参结果

- `k = 3`
- `beta = 1.0`
- `late_q = 0.0`
- `mean_inner_validation_mae = 2.9625`
- `std_inner_validation_mae = 1.2481`

这意味着：

- 在最终主线 `Baseline + KNN` 上，source-only inner-LOSO 选择直接落在 `beta=1.0`
- 因而当前终版主结果应明确切换为：
  - **`TimerXL + KNN (knn-only)`**

### 合法选参输出文件

- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_grid_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_stage_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_inner_best.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_global_summary.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/selected_knn_config_inner_loso.json`
- `paper_exec/train_only_param_selection/manifests/InnerLOSO选参汇总_RMS7_Baseline_E300.md`

### 合法选参图

- `paper_exec/figures/pure_rms7_baseline_knn_selection_heatmap.png`
- `paper_exec/figures/pure_rms7_baseline_knn_selection_heatmap.pdf`

### 与最终外层主结果的对应关系

由于合法选参已经明确选择：

- `beta = 1.0`

所以终版主线实际上不再是 `blend`，而是：

- **`delta-knn-only@k3, late_q=0.0`**

因此，后续论文主表应直接引用已经完成的 `Baseline + KNN @ epoch300, q=0.0` 的 `knn-only` 三折结果：

- `fold1`: `RMSE 2.6342 / MAE 2.2289`
- `fold2`: `RMSE 2.9668 / MAE 2.6133`
- `fold3`: `RMSE 3.1348 / MAE 2.5691`
- `avg`: `RMSE 2.9119 / MAE 2.4704`

也就是说，**在合法选参完成后，终版外层主结果的数值口径本身没有发生变化，只是参数来源被彻底合法化了。**

### 附录记录说明

为了后续论文附录完整披露，本轮合法选参已经完整保留了：

- 每个 inner task 的所有网格结果
- 每个 inner task 的 stage 指标
- 每个 inner task 的最优参数
- 全局汇总排序
- 最终固定配置 json

因此，后续附录不再需要回退到旧的 `TMA` 路径选参结果来解释终版主线参数来源。

## 状态更新（必须注意）

本文件保留了两类信息：

1. **历史阶段结果**
   - 基于 `TMA / retrieval backbone` 路径完成的 `inner-LOSO` 选参与对应结果
   - 这些结果对分析趋势有价值，但**不应再直接作为论文终版主线参数来源**

2. **当前终版方向**
   - 论文最终主线已经切换为：
     - `pure RMS7`
     - `epoch = 300`
     - `Baseline + KNN`
   - 因此，后续合法超参数选择必须在 **`Baseline + KNN`** 路径上重做

一句话：

> 本文件中凡是“基于 TMA 路径得到的 `k / beta / late_q`”结果，只能作为过渡分析，不应再直接作为论文终版唯一参数依据。论文终版参数来源，已经由上面的 `Baseline + KNN` 合法选参替代。

## 当前真正目标

在 `pure RMS7` 主线下，围绕最终模块链：

- `Baseline`
- `Baseline + KNN`

重新完成一次**完全合法**的 source-only inner-LOSO 选参。

## 当前最关键的后续动作

后续所有与论文终版参数有关的表述，应改成：

1. 先固定最终主线：
   - `pure RMS7`
   - `epoch = 300`
   - `Baseline + KNN`

2. 再在这条主线上重跑：
   - `k ∈ {3, 5, 7, 10}`
   - `beta ∈ {0.3, 0.5, 0.7, 1.0}`
   - `late_q ∈ {0.0, 0.5, 0.8}`

3. 然后用这套新结果替换本文档中旧的 `TMA` 路径选参结论

因此，本文件当前可用于：

- 回溯实验过程
- 保留阶段性分析依据

但论文终版真正应引用的选参结果，应来自后续新的 `Baseline + KNN inner-LOSO` 实验。

## 当前固定协议

- 特征：`pure RMS7`
- 标签：`mean`
- 训练协议：`no validation set`
- 固定轮数：`epoch=300`
- 外层评测：`LOCO`

## 本阶段顺序

1. 先补齐 `epoch=300` 的 `pure RMS7 + TMA` 三折
2. 再补齐 `epoch=300` 的 `pure RMS7 retrieval backbone` 三折
3. 然后在此基础上跑 `k × beta` source-only 选参

## 结果目录

### `pure RMS7 + TMA @ epoch300`

- `results/20260409_r7t300_fold1`
- `results/20260409_r7t300_fold2`
- `results/20260409_r7t300_fold3`
- `results/20260409_r7t300_fold1h`
- `results/20260409_r7t300_fold2h`
- `results/20260409_r7t300_fold3h`

### `pure RMS7 retrieval backbone @ epoch300`

- `results/20260409_r7r300_fold1`
- `results/20260409_r7r300_fold2`
- `results/20260409_r7r300_fold3`

## 当前状态

- `epoch` 敏感性已完成，`fold1` 上 `epoch=300` 为当前最优点
- 本文件当前先用于记录 `epoch=300` 的 backbone 补跑进度

## 已完成：pure RMS7 baseline @ epoch300

### 结果目录

- `results/20260409_r7e_fold1_e300`
- `results/20260409_r7e_fold2_e300`
- `results/20260409_r7e_fold3_e300`
- `results/20260409_r7e_fold1_e300h`
- `results/20260409_r7e_fold2_e300h`
- `results/20260409_r7e_fold3_e300h`

### head-only 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `5.0665` | `4.4243` |
| `fold2` | `6.2884` | `5.1470` |
| `fold3` | `5.7984` | `4.3575` |

### head-only 三折平均

- `avg RMSE = 5.7178`
- `avg MAE = 4.6429`

## 已完成：pure RMS7 + TMA @ epoch300

### 结果目录

- `results/20260409_r7t300_fold1`
- `results/20260409_r7t300_fold2`
- `results/20260409_r7t300_fold3`
- `results/20260409_r7t300_fold1h`
- `results/20260409_r7t300_fold2h`
- `results/20260409_r7t300_fold3h`

### head-only 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `5.9324` | `5.5669` |
| `fold2` | `5.6726` | `4.9189` |
| `fold3` | `5.2238` | `4.2480` |

### head-only 三折平均

- `avg RMSE = 5.6096`
- `avg MAE = 4.9113`

## 已完成：pure RMS7 retrieval backbone @ epoch300

### 结果目录

- `results/20260409_r7r300_fold1`
- `results/20260409_r7r300_fold2`
- `results/20260409_r7r300_fold3`

### checkpoint 路径

- `checkpoints/forecast_PHM_c1c4_to_c6_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu0_novalfix_r7r300_fold1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth`
- `checkpoints/forecast_PHM_c4c6_to_c1_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu1_novalfix_r7r300_fold2_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth`
- `checkpoints/forecast_PHM_c1c6_to_c4_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu2_novalfix_r7r300_fold3_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth`

## 下一步：source-only KNN 选参

### 第一轮固定设置

- `k ∈ {3,5,7,10,15}`
- `beta ∈ {0.0,0.3,0.5,0.7,1.0}`
- `late_q = 0.8`
- 选择协议：`inner leave-one-source-run-out`
- 选择指标：`mae_full_raw`

### 当前目的

- 判断 `pure RMS7 @ epoch300` 下最优 `beta` 是否接近 `1.0`
- 由此决定主结果更偏 `delta-blend` 还是 `delta-knn-only`

## 已完成：第一轮 inner-LOSO 选参

### 选参网格

- `k ∈ {3,5,7,10,15}`
- `beta ∈ {0.0,0.3,0.5,0.7,1.0}`
- `late_q = 0.8`
- 选择指标：`mae_full_raw`

### 全局最优固定参数

- `k = 3`
- `beta = 0.5`
- `late_q = 0.8`
- `mean_inner_validation_mae = 3.4568`
- `std_inner_validation_mae = 2.4007`

### 全局 summary 前五名

| Rank | k | beta | late_q | mean inner MAE | std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `3` | `0.5` | `0.8` | `3.4568` | `2.4007` |
| `2` | `5` | `0.5` | `0.8` | `3.4708` | `2.4031` |
| `3` | `7` | `0.5` | `0.8` | `3.4848` | `2.4048` |
| `4` | `10` | `0.5` | `0.8` | `3.5057` | `2.4050` |
| `5` | `15` | `0.5` | `0.8` | `3.5373` | `2.4020` |

### 各 inner task 最优

- `fold1:c1->c4`: `k=3`, `beta=0.0`, `late_q=0.8`, `metric=1.1317`
- `fold1:c4->c1`: `k=3`, `beta=0.0`, `late_q=0.8`, `metric=1.6854`
- `fold2:c4->c6`: `k=3`, `beta=1.0`, `late_q=0.8`, `metric=5.7971`
- `fold2:c6->c4`: `k=3`, `beta=0.5`, `late_q=0.8`, `metric=2.1296`
- `fold3:c1->c6`: `k=3`, `beta=1.0`, `late_q=0.8`, `metric=1.9379`
- `fold3:c6->c1`: `k=3`, `beta=0.5`, `late_q=0.8`, `metric=1.1143`

### 当前判断

- 全局最优 `beta` 停在 `0.5`，没有推到 `1.0`，说明 source-only 选参结果更偏 `blend` 而不是纯 `knn-only`。
- 所有 top-5 组合的 `beta` 都是 `0.5`，说明 `blend` 在 source-only 验证下相对稳定。
- `k` 的最优值落在最小邻居数 `3`，说明当前纯 `RMS7 @ epoch300` 更依赖局部最近邻，而不是更大的平均化邻域。

### 输出文件

- `paper_exec/train_only_param_selection/results_rms7_e300/inner_loso_knn_grid_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_e300/inner_loso_knn_stage_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_e300/inner_loso_knn_inner_best.csv`
- `paper_exec/train_only_param_selection/results_rms7_e300/inner_loso_knn_global_summary.csv`
- `paper_exec/train_only_param_selection/results_rms7_e300/selected_knn_config_inner_loso.json`
- `paper_exec/train_only_param_selection/manifests/InnerLOSO选参汇总_RMS7_E300.md`

## 已完成：pure RMS7 @ epoch300 三折最终 KNN 结果

### 结果目录

- `results/20260409_r7k300_fold1`
- `results/20260409_r7k300_fold2`
- `results/20260409_r7k300_fold3`

### 使用参数

- `k = 3`
- `beta = 0.5`
- `late_q = 0.8`

### delta-knn-only@k3 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `2.7946` | `2.4668` |
| `fold2` | `4.5432` | `4.4399` |
| `fold3` | `3.6844` | `3.3565` |

### delta-knn-only@k3 三折平均

- `avg RMSE = 3.6741`
- `avg MAE = 3.4211`

### delta-blend@k3_b05 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `4.0684` | `3.8396` |
| `fold2` | `3.8798` | `3.0113` |
| `fold3` | `4.1607` | `3.4610` |

### delta-blend@k3_b05 三折平均

- `avg RMSE = 4.0363`
- `avg MAE = 3.4373`

### 最终判断

- source-only 选参选择了 `beta=0.5`，说明从内层验证协议看，`blend` 是更合理的默认形式。
- 但在外层三折最终结果上，`delta-knn-only@k3` 的平均 MAE 略优于 `delta-blend@k3_b05`：`3.4211 < 3.4373`。
- 因此当前最稳的表述是：
  - `blend` 作为经 source-only 选参得到的默认集成形式；
  - `knn-only` 作为最终 strongest variant。

## 已完成：late_q 选择

### 固定条件

- `k = 3`
- `beta = 0.5`
- 仅扫描 `late_q ∈ {0.0, 0.5, 0.8}`
- 选择协议：`inner leave-one-source-run-out`
- 选择指标：`mae_full_raw`

### late_q 三档结果

| late_q | mean inner MAE | std | num inner tasks |
| --- | ---: | ---: | ---: |
| `0.0` | `3.0444` | `2.2751` | `6` |
| `0.5` | `3.1406` | `2.2353` | `6` |
| `0.8` | `3.4568` | `2.4007` | `6` |

### late_q 选择结论

- 当前纯 `RMS7 @ epoch300` 主线下，固定 `k=3`、`beta=0.5` 后，最优 `late_q` 为 `0.0`。
- 这说明在当前设定下，不再单独筛掉“早期库样本”反而更有利；使用全库进行残差检索更稳。
- 因此后续正式三折最终结果应更新为：
  - `k = 3`
  - `beta = 0.5`
  - `late_q = 0.0`

## 已完成：按 `late_q=0.0` 重跑三折最终评估

### 使用参数

- `k = 3`
- `beta = 0.5`
- `late_q = 0.0`

### delta-knn-only@k3 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `2.6889` | `2.3226` |
| `fold2` | `3.0900` | `2.7296` |
| `fold3` | `3.2093` | `2.6175` |

### delta-knn-only@k3 三折平均

- `avg RMSE = 2.9961`
- `avg MAE = 2.5566`

### delta-blend@k3_b05 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `3.9863` | `3.7592` |
| `fold2` | `3.8439` | `3.2150` |
| `fold3` | `4.1302` | `3.3501` |

### delta-blend@k3_b05 三折平均

- `avg RMSE = 3.9868`
- `avg MAE = 3.4414`

### 最终结论更新

- 将 `late_q` 从 `0.8` 更新到 `0.0` 后，`knn-only` 与 `blend` 都有改善，但 `knn-only` 的收益更大。
- 当前纯 `RMS7 @ epoch300` 主线下，最终最强结果是：
  - `delta-knn-only@k3`
  - `avg RMSE = 2.9961`
  - `avg MAE = 2.5566`
- 因此在最新固定配置下，`knn-only` 不再只是 strongest variant，而是已经明显优于 `blend`。

## 已完成：pure RMS7 baseline + KNN @ epoch300

### 说明

- 本组实验直接使用 `pure RMS7 baseline @ epoch300` 的 checkpoint 做检索评估。
- 固定参数与最终主线一致：
  - `k = 3`
  - `beta = 0.5`
  - `late_q = 0.0`

### delta-knn-only@k3 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `2.6342` | `2.2289` |
| `fold2` | `2.9668` | `2.6133` |
| `fold3` | `3.1348` | `2.5691` |

### delta-knn-only@k3 三折平均

- `avg RMSE = 2.9119`
- `avg MAE = 2.4704`

### delta-blend@k3_b05 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `3.0576` | `2.6248` |
| `fold2` | `4.1745` | `3.3863` |
| `fold3` | `4.3192` | `3.3879` |

### delta-blend@k3_b05 三折平均

- `avg RMSE = 3.8504`
- `avg MAE = 3.1330`

### 模块层面的直接结论

- `baseline + KNN` 明显优于单独 `+TMA`，说明当前主线的主要收益来自检索修正，而不是 TMA。
- `baseline + KNN` 甚至略优于 `TMA + KNN`：
  - `baseline + KNN knn-only avg MAE = 2.4704`
  - `TMA + KNN knn-only avg MAE = 2.5566`
- 因此在当前统一协议与最终主线特征下，TMA 不是必要模块，`TimerXL + KNN` 已经构成最强主线。
