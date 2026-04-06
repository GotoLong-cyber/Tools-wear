# Paper Exec Refresh TODO

更新时间：2026-04-01

## 最小线性执行清单

1. 检查 baseline clean 三折是否全部出现最终 `fullcurve_raw` 指标，并确认对应 checkpoint 存在。
2. 检查 TMA clean 三折是否全部出现最终 `fullcurve_raw` 指标，并确认对应 checkpoint 存在。
3. 执行 clean retrieval inference，要求三折都基于 clean retrieval-backbone checkpoint、固定超参数和 train-only library。
4. 刷新 `paper_exec` 下全部 CSV、图、表、caption、zh/en snippets，并确保不混入旧 blocked 结果。
5. 复核 execution_log、clean_run_manifest、clean_checkpoint_index、paper_exec_manifest、paper_asset_dashboard 的一致性。
6. 改判：仅当 clean 三折结果齐全、clean retrieval inference 齐全、`paper_exec` 已完整刷新时，才把对应资产从 blocked 改为 formal-ready。

## 当前阻塞点

- baseline clean 未齐
- TMA clean 未齐

## 下一步唯一动作

先补齐 baseline clean 与 TMA clean 的缺失折次，其他动作一律后置。
