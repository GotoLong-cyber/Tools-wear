# full133 新协议进度

## 目标

在统一新协议下重跑 `full133` 三折标准基线，为后续时间对齐与 KNN 消融建立可辩护的特征基准。

统一协议：

- 标签：`wear_agg = mean`
- 训练：`no validation set`
- 训练轮数：`fixed epoch = 200`
- 外层评测：三折 `LOCO`
- 当前阶段：仅跑 `full133` 基线，不加 TMA，不加 KNN

## 命名约定

为避免与既有结果混淆，本轮采用单独、短命名结果目录：

- `20260408_f133_f1`
- `20260408_f133_f2`
- `20260408_f133_f3`

对应 head-only 评估目录：

- `20260408_f133h_f1`
- `20260408_f133h_f2`
- `20260408_f133h_f3`

运行时目录：

- `runtime_f133_fold1_g1`
- `runtime_f133_fold2_g2`
- `runtime_f133_fold3_g3`

## 当前状态

- [x] 已确认 `full133` 原始特征文件位于 `dataset/passlevel_full133_npz`
- [x] 已将 `f133` 接入统一运行入口 `run_clean_timer_fold.sh`
- [x] 已确定本轮仅作为标准基线，不混入 TMA / KNN
- [x] fold1 已启动（session `77222`）
- [x] fold2 已启动（session `41935`）
- [x] fold3 已启动（session `70849`）
- [ ] 三折训练完成
- [ ] 三折 head-only 评估完成
- [ ] 汇总写入总结果 md

## 备注

- 本轮 `full133` 的主要作用是建立“统一协议下的标准特征基线”，而不是直接作为最终最优结果。
- 跑完之后，下一步再决定是否在 `full133` 基线上叠加时间对齐与 KNN。
- 本轮三折结果目录：
  - `results/20260408_f133_f1`
  - `results/20260408_f133_f2`
  - `results/20260408_f133_f3`
- 本轮三折评估目录：
  - `results/20260408_f133h_f1`
  - `results/20260408_f133h_f2`
  - `results/20260408_f133h_f3`
