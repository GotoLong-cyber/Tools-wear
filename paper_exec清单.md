# 论文执行资产清单

## 已生成资产

- `paper_exec/csv/`：机器可读的汇总 CSV
- `paper_exec/figures/`：可直接用于投稿的 PNG/PDF 图件
- `paper_exec/tables/`：Markdown 表格草稿
- `paper_exec/captions/`：图注/表注草稿
- `paper_exec/结果片段_中文.md`、`paper_exec/结果片段_英文.md`：结果描述片段
- `paper_exec/论文资产看板.md`：资产就绪看板

## 正式结果与探索结果的划分

- 正式主线：TimerXL backbone -> TMA 辅助增强 -> KNN-DRR 主创新
- 已弃用 / 探索性分支：stage-align、gated retrieval、test-fold hyperparameter search
- 已弃用 / 仅作参考的结果目录：
  - `results/20260401_RetrievalV21_cleanformal`
  - `results/20260401_RetrievalV21_trainonlyselect`
  - `results/20260330_RetrievalV21_fixedcfg_rerun`

## 协议状态

- `retrieval hyperparameters fixed`：已固定
- `inner LOSO train-only selection`：已启用
- `gated retrieval excluded`：已排除
- `stage-align excluded`：已排除
- `legacy code still contains test-eval branch`：仍存在
- `formal clean wrappers disable train-time test evaluation`：已生效

当前结论：当前正式可用主结果已经来自 clean checkpoint 与 inner-LOSO 选出的固定检索参数（`k=10, beta=0.7, late_q=0.0`）。可选比较表仍未完全补齐，但主证据链已不再受训练协议问题阻塞。

附录增强资产：

- `paper_exec/csv/appendix_inner_loso_sensitivity.csv`
- `paper_exec/tables/表A2_InnerLOSO敏感性.md`
- `paper_exec/figures/FigA1_inner_loso_sensitivity.*`
