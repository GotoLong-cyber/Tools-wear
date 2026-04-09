# RMS7_PTP7_小波能量进度

## 统一协议

- 标签聚合：`wear_agg = mean`
- 训练协议：`no validation set`
- 固定训练轮数：`epoch = 200`
- 外层评测：`LOCO`
- 当前阶段目标：先完成 `RMS7 + PTP7 + wavelet energy` baseline 三折

## 特征定义

- `RMS7`：7 个通道 RMS
- `PTP7`：7 个通道 peak-to-peak / range
- `wavelet energy ratio 7`：对每个通道原始 pass 信号做 `db1` 三层小波分解，取 `detail energy / total energy`
- 总维度：`21`

## 结果目录

- 训练：
  - `results/20260408_r7p7w_fold1`
  - `results/20260408_r7p7w_fold2`
  - `results/20260408_r7p7w_fold3`
- head-only 评估：
  - `results/20260408_r7p7wh_fold1`
  - `results/20260408_r7p7wh_fold2`
  - `results/20260408_r7p7wh_fold3`
- `+TMA` 训练：
  - `results/20260408_r7p7wt_fold1`
  - `results/20260408_r7p7wt_fold2`
  - `results/20260408_r7p7wt_fold3`
- `+TMA` head-only 评估：
  - `results/20260408_r7p7wth_fold1`
  - `results/20260408_r7p7wth_fold2`
  - `results/20260408_r7p7wth_fold3`
- `+TMA+KNN` retrieval：
  - `results/20260408_r7p7wr_fold1`
  - `results/20260408_r7p7wr_fold2`
  - `results/20260408_r7p7wr_fold3`
- `+TMA+KNN` 指标：
  - `results/20260408_r7p7wk_fold1`
  - `results/20260408_r7p7wk_fold2`
  - `results/20260408_r7p7wk_fold3`

## baseline 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `12.4688` | `8.5399` |
| `fold2` | `5.1801` | `4.7580` |
| `fold3` | `6.8544` | `5.4470` |

### baseline 三折平均

- `avg RMSE = 8.1677`
- `avg MAE = 6.2483`

## +TMA 三折结果

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `10.6094` | `7.1976` |
| `fold2` | `5.0962` | `4.7034` |
| `fold3` | `5.0260` | `3.9782` |

### +TMA 三折平均

- `avg RMSE = 6.9105`
- `avg MAE = 5.2931`

## +TMA+KNN 三折结果

### delta-knn-only@k5

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `2.9379` | `2.6119` |
| `fold2` | `4.7091` | `4.6073` |
| `fold3` | `3.8085` | `3.5049` |

### delta-knn-only@k5 三折平均

- `avg RMSE = 3.8185`
- `avg MAE = 3.5747`

### delta-blend@k5_b05

| Fold | RMSE | MAE |
| --- | ---: | ---: |
| `fold1` | `4.9437` | `3.7868` |
| `fold2` | `4.7188` | `4.6554` |
| `fold3` | `4.0782` | `3.5755` |

### delta-blend@k5_b05 三折平均

- `avg RMSE = 4.5802`
- `avg MAE = 4.0059`

## 当前观察

- 这条 `21` 维组合线上，`TMA` 首次表现出相对明确的独立增益：`avg MAE 6.2483 -> 5.2931`。
- 在此基础上，`KNN` 仍然能继续显著压低误差，其中平均上 `delta-knn-only` 优于 `delta-blend`。
- 但即便加入 `TMA` 和 `KNN`，这条组合线的平均结果仍未超过 `pure RMS7 + TMA + KNN`。
