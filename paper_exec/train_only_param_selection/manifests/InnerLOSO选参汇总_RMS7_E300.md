# Inner LOSO Selection Summary (pure RMS7 @ epoch300)

- 候选 `k`: `[3]`
- 候选 `beta`: `[0.5]`
- 候选 `late_q`: `[0.0, 0.5, 0.8]`
- 选择指标: `mae_full_raw`

## 每个 inner task 最优

- fold1:c1->c4: k=3, beta=0.5, late_q=0.0, metric=1.8177
- fold1:c4->c1: k=3, beta=0.5, late_q=0.0, metric=2.8097
- fold2:c4->c6: k=3, beta=0.5, late_q=0.0, metric=8.0182
- fold2:c6->c4: k=3, beta=0.5, late_q=0.0, metric=1.2885
- fold3:c1->c6: k=3, beta=0.5, late_q=0.0, metric=2.4228
- fold3:c6->c1: k=3, beta=0.5, late_q=0.8, metric=1.1143

## 全局固定参数

- k=3
- beta=0.5
- late_q=0.0
- mean_inner_validation_mae=3.0444
- std_inner_validation_mae=2.2751

## 输出文件

- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_e300_lateq/inner_loso_knn_grid_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_e300_lateq/inner_loso_knn_stage_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_e300_lateq/inner_loso_knn_inner_best.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_e300_lateq/inner_loso_knn_global_summary.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_e300_lateq/selected_knn_config_inner_loso.json`
