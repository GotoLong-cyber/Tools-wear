# Paper Figures

This directory contains paper-ready figures copied or generated from the final experiment artifacts.

| Category | Filename | Title | Purpose | Source |
| --- | --- | --- | --- | --- |
| `01_problem_setup` | `Fig01_A1_C1C4C6_TimeScale_CurveComparison.png` | C1/C4/C6 wear curve comparison | Motivates cross-condition time-scale mismatch. | `feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_c1c4c6_curve_compare.png` |
| `01_problem_setup` | `Fig02_A1_C1C4C6_NormalizedProgressCurves.png` | Normalized progress wear curves | Shows normalized trajectory mismatch across tools. | `feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_wear_curves_normalized_progress.png` |
| `01_problem_setup` | `Fig03_A1_C1C4C6_StageSlopeComparison.png` | Stage slope comparison | Shows stage-wise degradation slope differences. | `feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_stage_slopes.png` |
| `02_method` | `Fig04_A2_Baseline_vs_A2_Structure.svg` | Baseline vs A2 structure | Method-side structure comparison for A2 augmentation stage. | `feature_alignment_diagnosis/outputs/A2_dynstride_diagnostics_20260323_1743/baseline_vs_a2final_structure.svg` |
| `03_main_results` | `Fig05_RetrievalV21_Fold1_CurrentBest_vs_Retrieval.png` | Fold1 current-best vs Retrieval V2.1 | Main result curve for hardest fold. | `results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold1.png` |
| `03_main_results` | `Fig06_RetrievalV21_Fold2_CurrentBest_vs_Retrieval.png` | Fold2 current-best vs Retrieval V2.1 | Main result curve for fold2. | `results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold2.png` |
| `03_main_results` | `Fig07_RetrievalV21_Fold3_CurrentBest_vs_Retrieval.png` | Fold3 current-best vs Retrieval V2.1 | Main result curve for fold3. | `results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold3.png` |
| `04_mechanism` | `Fig09_Fold1_CurrentBest_StageCurve.png` | Fold1 stage-colored wear curve | Highlights early/mid/late partition on current-best fold1. | `feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_currentbest/fold1_stage_curve_colored.png` |
| `04_mechanism` | `Fig10_Fold1_CurrentBest_StageResiduals.png` | Fold1 stage residual curve | Shows late-stage systematic underestimation before retrieval correction. | `feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_currentbest/fold1_stage_residuals.png` |
| `03_main_results` | `Fig08_RetrievalV21_ThreeFold_MAE_RMSE.png` | Retrieval V2.1 three-fold MAE/RMSE comparison | Main quantitative result figure for paper results section. | `generated from results/20260325_RetrievalV21_formal/knn_delta_fold*_overall_metrics.csv` |
| `04_mechanism` | `Fig11_Fold1_StageRepair_MAE_Residual.png` | Fold1 stage repair by Retrieval V2.1 | Mechanism figure showing late-stage MAE and residual reduction. | `generated from results/20260325_RetrievalV21_formal/knn_delta_fold1_stage_metrics.csv` |
| `05_ablation` | `Fig12_RetrievalV21_Sensitivity_K_q_beta.png` | Retrieval V2.1 sensitivity to K, q, and beta | Ablation figure showing robustness to key retrieval hyperparameters. | `generated from formal/sensitivity three-fold overall metrics` |
