# RMS7_PTP7新协议进度

## 统一协议

- 标签聚合：`wear_agg = mean`
- 训练协议：`no validation set`
- 固定训练轮数：`epoch = 200`
- 外层评测：`LOCO`
- 当前阶段目标：先完成 `RMS7 + PTP7 baseline` 三折

## 特征定义

- `RMS7`：`Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27`
- `PTP7`：`Feat_4, Feat_8, Feat_12, Feat_16, Feat_20, Feat_24, Feat_28`

## 计划目录

- `results/20260408_r7p7b_fold1`
- `results/20260408_r7p7b_fold2`
- `results/20260408_r7p7b_fold3`
- `results/20260408_r7p7bh_fold1`
- `results/20260408_r7p7bh_fold2`
- `results/20260408_r7p7bh_fold3`

## 执行状态

- [x] `RMS7 + PTP7` 生成脚本已补齐
- [x] fold1 baseline 启动并完成
- [x] fold2 baseline 启动并完成
- [x] fold3 baseline 启动并完成
- [x] baseline 三折完成
- [ ] 结果汇总写回

## 已完成：RMS7 + PTP7 baseline（head-only）

### 结果目录

- `results/20260408_r7p7b_fold1`
- `results/20260408_r7p7b_fold2`
- `results/20260408_r7p7b_fold3`
- `results/20260408_r7p7bh_fold1`
- `results/20260408_r7p7bh_fold2`
- `results/20260408_r7p7bh_fold3`

### 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1 (c1+c4 -> c6)` | `7.8411` | `5.3528` |
| `fold2 (c4+c6 -> c1)` | `5.8455` | `5.0482` |
| `fold3 (c1+c6 -> c4)` | `4.4019` | `3.6138` |

### 三折平均

- `head-only avg RMSE = 6.0295`
- `head-only avg MAE = 4.6716`
