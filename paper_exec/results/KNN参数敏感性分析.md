# KNN 参数敏感性分析

## 合法选参口径

- 主线：`Pure RMS7 + Baseline + KNN`
- 训练：`epoch = 300`
- 内层协议：`source-only inner-LOSO`
- 外层测试未参与选参

## 最终合法参数

- `k = 3`
- `beta = 1.0`
- `late_q = 0.0`

## 主文图

- 热力图：
  - `paper_exec/figures/pure_rms7_baseline_knn_selection_heatmap.png`
  - `paper_exec/figures/pure_rms7_baseline_knn_selection_heatmap.pdf`
- k 敏感性曲线（固定 `beta=1.0, late_q=0.0`，外层 LOCO 三折）：
  - `paper_exec/figures/knn_sensitivity_curve.png`
  - `paper_exec/figures/knn_sensitivity_curve.pdf`
- 外层 LOCO 正式附录图（RMS7）：
  - `paper_exec/figures/FigA2_outer_loco_k_sweep.png`
  - `paper_exec/figures/FigA2_outer_loco_k_sweep.pdf`
- 外层 LOCO 正式附录图（Full-133）：
  - `paper_exec/figures/FigA3_outer_loco_k_sweep_full133.png`
  - `paper_exec/figures/FigA3_outer_loco_k_sweep_full133.pdf`

## 合法内层选参结果

固定 `beta=1.0, late_q=0.0` 时，内层 `source-only inner-LOSO` 的 `k` 曲线如下：

| k | mean inner MAE | std | num inner tasks |
|---|---------------:|----:|----------------:|
| 3 | 2.9625 | 1.2481 | 6 |
| 5 | 2.9676 | 1.2513 | 6 |
| 7 | 2.9722 | 1.2541 | 6 |
| 10 | 2.9786 | 1.2576 | 6 |

因此，**论文终版合法参数仍应采用 `k=3`**。这是因为最终主线参数必须来自源域内部选参，而不能根据外层测试结果反调。

## 外层 LOCO 的 k 扫描（附录保留）

为补充附录中的敏感性分析，我们在最终主线 `Pure RMS7 + Baseline + KNN` 下，固定：

- `beta = 1.0`
- `late_q = 0.0`
- `epoch = 300`

并额外对外层 LOCO 三折执行了 `k ∈ {1, 3, 5, 7, 9, 11, 15}` 的完整扫描。

| k | Fold1 MAE | Fold2 MAE | Fold3 MAE | 平均 MAE |
|---|----------:|----------:|----------:|---------:|
| 1  | 2.1701 | 2.6568 | 2.5618 | 2.4629 |
| 3  | 2.2288 | 2.6132 | 2.5692 | 2.4704 |
| 5  | 2.2378 | 2.5908 | 2.5637 | 2.4641 |
| 7  | 2.2349 | 2.4904 | 2.5636 | 2.4296 |
| 9  | 2.2492 | 2.3997 | 2.5684 | 2.4058 |
| 11 | 2.2619 | 2.2990 | 2.5717 | 2.3775 |
| 15 | 2.2567 | 2.1902 | 2.5857 | 2.3442 |

对应 CSV：

- `paper_exec/results/knn_outer_k_sweep_rms7_baseline_e300.csv`
- `paper_exec/results/K外层扫描实验_RMS7_Baseline_E300.csv`
- `paper_exec/tables/K外层扫描实验_RMS7_Baseline_E300.md`

## Full-133 的外层 LOCO k 扫描（附录补充）

为补充对比，我们在 `Full-133 + Baseline + KNN` 下，固定：

- `beta = 1.0`
- `late_q = 0.0`
- `epoch = 300`

并额外对外层 LOCO 三折执行了 `k ∈ {1, 3, 5, 7, 9, 11, 15}` 的完整扫描。

| k | Fold1 MAE | Fold2 MAE | Fold3 MAE | 平均 MAE |
|---|----------:|----------:|----------:|---------:|
| 1  | 2.4284 | 4.4430 | 3.4236 | 3.4317 |
| 3  | 2.4263 | 4.4329 | 3.4432 | 3.4342 |
| 5  | 2.4228 | 4.4045 | 3.4546 | 3.4273 |
| 7  | 2.4179 | 4.3634 | 3.5812 | 3.4542 |
| 9  | 2.4152 | 4.5987 | 3.5603 | 3.5248 |
| 11 | 2.4110 | 4.7403 | 3.6320 | 3.5944 |
| 15 | 2.4020 | 4.9135 | 3.7257 | 3.6804 |

对应 CSV / Markdown：

- `paper_exec/results/K外层扫描实验_Full133_Baseline_E300.csv`
- `paper_exec/tables/K外层扫描实验_Full133_Baseline_E300.md`

## 如何解释“内层选参”和“外层扫 k”不一致

- **用于论文主结果的合法参数** 必须来自源域内部的 `inner-LOSO` 选参，因此最终主线仍写作：
  - `k=3, beta=1.0, late_q=0.0`
- **外层 k 扫描** 只用于补充敏感性分析与附录展示，说明在最终主线下，性能对更大邻居数并不敏感，且在当前三折上存在继续下降的趋势。
- 因此，这组结果不能反过来改写主结果参数，只能作为：
  - 模型鲁棒性的补充证据
  - `k` 选择并非极端敏感的附录说明

## 完整搜索空间（附录保留）

- `k ∈ {3, 5, 7, 10}`
- `beta ∈ {0.3, 0.5, 0.7, 1.0}`
- `late_q ∈ {0.0, 0.5, 0.8}`

完整记录：
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_grid_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_stage_all.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_inner_best.csv`
- `paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_global_summary.csv`

## 结论

- 合法主线下的最优参数为 `k=3, beta=1.0, late_q=0.0`。
- 内层合法选参表明：在 `source-only inner-LOSO` 下，`k=3` 是最佳选择。
- 外层 LOCO 的补充 `k` 扫描表明：在固定 `beta=1.0, late_q=0.0` 时，性能对更大 `k` 不敏感，且在当前三折上存在继续改善趋势。
- `Full-133` 的外层 `k` 扫描则说明：高维统计特征接入 KNN 后仍可被有效利用，但其对 `k` 的外层表现整体仍弱于 `Pure RMS7` 主线。
- 因此，论文主结果应继续采用合法参数 `k=3`；而外层 `k` 扫描作为附录中的稳健性分析保留。
