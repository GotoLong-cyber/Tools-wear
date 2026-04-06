# Inner LOSO 选参摘要

- 候选 `k`: `[3, 5, 10]`
- 候选 `beta`: `[0.3, 0.5, 0.7]`
- 候选 `late_q`: `[0.0, 0.8]`
- 选择指标: `mae_full_raw`

## 每个 inner task 的最优配置

- fold1:c1->c4: k=10, beta=0.3, late_q=0.0, metric=1.9078
- fold1:c4->c1: k=10, beta=0.3, late_q=0.0, metric=1.8396
- fold2:c4->c6: k=3, beta=0.7, late_q=0.0, metric=5.9710
- fold2:c6->c4: k=3, beta=0.7, late_q=0.0, metric=1.4784
- fold3:c1->c6: k=10, beta=0.7, late_q=0.8, metric=4.2408
- fold3:c6->c1: k=10, beta=0.3, late_q=0.0, metric=1.9381

## 全局固定参数

- k=10
- beta=0.7
- late_q=0.0
- mean_inner_validation_mae=3.3114
- std_inner_validation_mae=1.5638

## 输出文件

- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_knn_grid_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_knn_stage_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_knn_inner_best.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_knn_global_summary.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/selected_knn_config_inner_loso.json`

## 新增补充资产

以下文件用于补足此前未单独展示的 inner LOSO 曲线与 `MAE/RMSE`：

- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_task_metrics.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_task_stage_metrics.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/manifests/InnerLOSO曲线资产汇总.md`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_*.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_*.pdf`
