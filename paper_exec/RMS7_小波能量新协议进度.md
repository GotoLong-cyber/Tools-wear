# RMS7_小波能量新协议进度

## 统一协议

- 标签聚合：`wear_agg = mean`
- 训练协议：`no validation set`
- 固定训练轮数：`epoch = 200`
- 外层评测：`LOCO`
- 当前阶段目标：先完成 `RMS7 + wavelet energy` baseline 三折

## 特征定义

- `RMS7`：`Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27`
- `wavelet energy ratio`：对每个通道原始 pass 信号做 `db1` 三层离散小波分解，取 `detail energy / total energy`
- 总维度：`14`

## 结果目录

- 训练：
  - `results/20260408_r7w_fold1`
  - `results/20260408_r7w_fold2`
  - `results/20260408_r7w_fold3`
- head-only 评估：
  - `results/20260408_r7wh_fold1`
  - `results/20260408_r7wh_fold2`
  - `results/20260408_r7wh_fold3`

## 当前状态

- 已接入特征生成脚本与 baseline 训练脚本
- baseline 三折已启动：
  - `fold1`: session `89170`
  - `fold2`: session `72723`
  - `fold3`: session `46072`
- 等待三折 baseline 结果
