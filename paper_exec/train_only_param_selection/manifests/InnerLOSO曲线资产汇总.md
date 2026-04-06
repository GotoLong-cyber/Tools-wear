# Inner LOSO 曲线与指标补充说明

本文件补充 `inner LOSO` 选参协议下缺失的两类资产：
1. 每个 inner task 在最终全局固定参数下的 MAE/RMSE 汇总
2. 每个 inner task 的真实曲线、head-only 曲线与 global-selected 曲线图

- 全局固定参数：`k=10, beta=0.7, late_q=0.0`
- 指标文件：`/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_task_metrics.csv`
- 分阶段文件：`/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results/inner_loso_task_stage_metrics.csv`

## 每个 inner task 的全局配置结果

- fold1:c1->c4: MAE=2.2442, RMSE=3.0466, library_run=c1, query_run=c4
- fold1:c4->c1: MAE=3.3376, RMSE=3.9653, library_run=c4, query_run=c1
- fold2:c4->c6: MAE=5.9983, RMSE=6.5946, library_run=c4, query_run=c6
- fold2:c6->c4: MAE=1.5373, RMSE=1.8108, library_run=c6, query_run=c4
- fold3:c1->c6: MAE=4.6325, RMSE=5.9882, library_run=c1, query_run=c6
- fold3:c6->c1: MAE=2.1190, RMSE=2.8773, library_run=c6, query_run=c1

## 曲线图文件

- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold1_c1_to_c4.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold1_c4_to_c1.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold2_c4_to_c6.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold2_c6_to_c4.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold3_c1_to_c6.png`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/figures/inner_loso_curve_fold3_c6_to_c1.png`
