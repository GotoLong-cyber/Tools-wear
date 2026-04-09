# Inner LOSO Selection Summary (pure RMS7 baseline+KNN @ epoch300)

- 候选 `k`: `[3, 5, 7, 10]`
- 候选 `beta`: `[0.3, 0.5, 0.7, 1.0]`
- 候选 `late_q`: `[0.0, 0.5, 0.8]`
- 选择指标: `mae_full_raw`

## 每个 inner task 最优

- fold1:c1->c4: k=3, beta=0.3, late_q=0.0, metric=1.5870
- fold1:c4->c1: k=5, beta=0.3, late_q=0.0, metric=2.6277
- fold2:c4->c6: k=3, beta=1.0, late_q=0.0, metric=5.0818
- fold2:c6->c4: k=5, beta=0.7, late_q=0.0, metric=1.2163
- fold3:c1->c6: k=3, beta=1.0, late_q=0.0, metric=1.7398
- fold3:c6->c1: k=3, beta=0.3, late_q=0.8, metric=1.5840

## 全局固定参数

- k=3
- beta=1.0
- late_q=0.0
- mean_inner_validation_mae=2.9625
- std_inner_validation_mae=1.2481

## 输出文件

- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_grid_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_stage_all.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_inner_best.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_baseline_e300/inner_loso_knn_global_summary.csv`
- `/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/train_only_param_selection/results_rms7_baseline_e300/selected_knn_config_inner_loso.json`
