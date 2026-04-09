# 纯RMS7新协议进度

## 统一协议

- 标签聚合：`wear_agg = mean`
- 训练协议：`no validation set`
- 固定训练轮数：`epoch = 200`
- 外层评测：`LOCO`
- 特征定义：纯 `RMS7`，即 `Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27`

## 结果目录规划

### baseline

- `results/20260408_r7b_f1`
- `results/20260408_r7b_f2`
- `results/20260408_r7b_f3`
- `results/20260408_r7bh_f1`
- `results/20260408_r7bh_f2`
- `results/20260408_r7bh_f3`

### TMA

- `results/20260408_r7t_f1`
- `results/20260408_r7t_f2`
- `results/20260408_r7t_f3`
- `results/20260408_r7th_f1`
- `results/20260408_r7th_f2`
- `results/20260408_r7th_f3`

### retrieval

- `results/20260408_r7r_f1`
- `results/20260408_r7r_f2`
- `results/20260408_r7r_f3`
- `results/20260408_r7k_f1`
- `results/20260408_r7k_f2`
- `results/20260408_r7k_f3`

## 执行状态

- [x] 纯 `RMS7` 生成脚本已补齐
- [x] fold1 任务启动并完成 baseline
- [x] fold2 任务启动并完成 baseline
- [x] fold3 任务启动并完成 baseline
- [x] baseline 三折完成
- [x] TMA 三折完成
- [x] retrieval 三折完成
- [ ] 结果汇总写回

## 已完成：pure RMS7 baseline（head-only）

### 结果目录

- `results/20260408_r7b_fold1`
- `results/20260408_r7b_fold2`
- `results/20260408_r7b_fold3`
- `results/20260408_r7bh_fold1`
- `results/20260408_r7bh_fold2`
- `results/20260408_r7bh_fold3`

### 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1 (c1+c4 -> c6)` | `5.5836` | `5.2127` |
| `fold2 (c4+c6 -> c1)` | `4.8488` | `4.1546` |
| `fold3 (c1+c6 -> c4)` | `5.1337` | `3.7970` |

### 三折平均

- `head-only avg RMSE = 5.1887`
- `head-only avg MAE = 4.3881`

## 已完成：pure RMS7 + TMA

### 结果目录

- `results/20260408_r7t_fold1`
- `results/20260408_r7t_fold2`
- `results/20260408_r7t_fold3`
- `results/20260408_r7th_fold1`
- `results/20260408_r7th_fold2`
- `results/20260408_r7th_fold3`

### head-only 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1 (c1+c4 -> c6)` | `6.4292` | `5.7977` |
| `fold2 (c4+c6 -> c1)` | `4.9371` | `4.1570` |
| `fold3 (c1+c6 -> c4)` | `4.1966` | `3.5394` |

### 三折平均

- `head-only avg RMSE = 5.1876`
- `head-only avg MAE = 4.4981`

## 已完成：pure RMS7 + TMA + KNN

### 结果目录

- `results/20260408_r7r_fold1`
- `results/20260408_r7r_fold2`
- `results/20260408_r7r_fold3`
- `results/20260408_r7k_fold1`
- `results/20260408_r7k_fold2`
- `results/20260408_r7k_fold3`

### delta-knn-only@k5 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `3.0424` | `2.6900` |
| `fold2` | `4.7076` | `4.6059` |
| `fold3` | `3.5913` | `3.2820` |

### delta-knn-only@k5 三折平均

- `avg RMSE = 3.7804`
- `avg MAE = 3.5260`

### delta-blend@k5_b05 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `4.4303` | `4.0788` |
| `fold2` | `3.8091` | `3.1766` |
| `fold3` | `3.6591` | `3.1157` |

### delta-blend@k5_b05 三折平均

- `avg RMSE = 3.9662`
- `avg MAE = 3.4570`
