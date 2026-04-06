# Few-Shot Cross-Condition Tool Wear Prediction via Train-Time Multi-Stride Augmentation and KNN Delta Residual Retrieval

## Abstract
Few-shot cross-condition tool wear prediction is difficult because degradation rates differ across cutting conditions, so fixed-step temporal representations are often misaligned across domains. In this setting, a strong time-series backbone alone can still produce a structural late-stage underestimation of future wear increments. We build a three-level pipeline on PHM2010: TimerXL as the backbone predictor, Train-time Multi-stride Augmentation (TMA) as an auxiliary representation enhancement, and KNN-based Delta Residual Retrieval (KNN-DRR) as the primary correction module. TMA is applied only during training and does not use any test-condition information; KNN-DRR performs residual correction at inference with a train-only retrieval library. Hyperparameters are fixed by inner leave-one-source-out validation within source conditions, and each outer test fold is evaluated once. Under three-fold leave-one-condition-out evaluation, average MAE is reduced from 8.7711 (TimerXL head-only) to 6.1484 (TimerXL+TMA) and further to 3.1784 (TimerXL+TMA+KNN-DRR); average RMSE is reduced from 11.9747 to 9.4630 and further to 3.9763. On the representative hardest fold (c1,c4 -> c6), fold-level MAE decreases from 13.3984 to 9.2761 and further to 3.2065. Late-stage MAE on this fold drops from 21.2180 to 5.7781 with delta-blend correction. These results support a clear division of roles: TimerXL provides the backbone trend, TMA improves cross-condition comparability, and KNN-DRR performs the main residual correction.

## 1. Introduction
Tool wear prediction in real manufacturing is often constrained by limited labeled trajectories under each operating condition. In cross-condition deployment, the mismatch is not only in feature distribution but also in degradation tempo: the same index in time does not correspond to the same wear stage across conditions. This temporal-scale mismatch is especially damaging in late wear stages, where future increments accelerate and standard predictors tend to under-estimate.

Our starting point is that foundation-style time-series backbones are useful but not sufficient in this regime. We therefore separate the solution into three layers rather than treating all components as equally novel. First, TimerXL provides the baseline sequential representation and trend prediction. Second, TMA introduces train-time multi-stride augmentation to improve representation comparability under rate variation. Third, KNN-DRR performs explicit residual correction on future wear increments using a train-only retrieval library.

This design targets the structural failure mode directly while preserving a strict clean protocol: no outer-test tuning, no test-label leakage, and a globally fixed retrieval configuration selected only by inner train-only validation.

### Contributions
1. We present a formal evidence chain for few-shot cross-condition tool wear prediction on PHM2010 with a clear module hierarchy: TimerXL backbone, TMA auxiliary enhancement, and KNN-DRR primary innovation.
2. We show that train-time temporal-scale enhancement and retrieval-based residual correction provide complementary gains, reducing average MAE from 8.7711 to 3.1784 under clean protocol constraints.
3. We provide stage-level evidence that the dominant structural error is late-stage underestimation of future wear increments, and that residual correction substantially mitigates this bias.

## 2. Method
### 2.1 Problem Setup
Given a history window from source conditions, the model predicts future wear values under an unseen target condition. The key difficulty is cross-condition degradation-rate mismatch: equal time indices can represent different physical wear stages.

### 2.2 TimerXL Backbone
TimerXL is used as the backbone predictor to model temporal dynamics and produce baseline future wear trends. In our framing, TimerXL is the platform model rather than the primary innovation.

### 2.3 Train-Time Multi-Stride Augmentation (TMA)
TMA is an auxiliary training-time mechanism. It perturbs temporal stride in training windows to improve representation comparability when degradation tempo differs across conditions. TMA is not an inference-time alignment module and is not applied adaptively on validation or test data.

### 2.4 KNN Delta Residual Retrieval (KNN-DRR)
KNN-DRR is the primary correction module. At inference, it retrieves similar historical trajectories from a train-only library and corrects future wear increment residuals on top of backbone trends. This is a residual-correction design, not a replacement of the backbone.

### 2.5 Train-Only Hyperparameter Protocol
Retrieval hyperparameters are fixed by inner leave-one-source-out validation inside source conditions. For each outer fold, source conditions are alternated as library/validation domains, and errors are averaged across inner splits. The globally fixed configuration is then evaluated once on each outer test fold. Final formal setting: k=10, beta=0.7, late_q=0.0.

## 3. Experimental Setup
### 3.1 Dataset and Cross-Condition Protocol
We evaluate on PHM2010 with three conditions (c1, c4, c6) under three-fold leave-one-condition-out (LOCO):
- fold1: c1,c4 -> c6
- fold2: c4,c6 -> c1
- fold3: c1,c6 -> c4

### 3.2 Evaluation Metrics
Primary metrics are full-curve MAE and RMSE (um). For mechanism analysis, we additionally inspect representative hardest-fold behavior and stage-wise residual statistics (early/mid/late).

### 3.3 Compared Variants
We report a formal three-stage chain:
- TimerXL head-only
- TimerXL + TMA
- TimerXL + TMA + KNN-DRR

All reported main results follow clean protocol with train-only inner LOSO tuning.

## 4. Results
### 4.1 Main Three-Fold Results
Average performance over three outer folds:
- TimerXL head-only: MAE 8.7711, RMSE 11.9747
- TimerXL + TMA: MAE 6.1484, RMSE 9.4630
- TimerXL + TMA + KNN-DRR: MAE 3.1784, RMSE 3.9763

This chain indicates two complementary gains: TMA improves cross-condition representation usability, and KNN-DRR delivers the major residual correction gain.

### 4.2 Representative Hardest Fold
On fold1 (c1,c4 -> c6), fold-level MAE changes from:
- 13.3984 (head-only)
- to 9.2761 (TimerXL+TMA)
- to 3.2065 (TimerXL+TMA+KNN-DRR)

This fold exposes the strongest late-stage mismatch and highlights the value of explicit residual correction.

### 4.3 Stage-Wise Bias Evidence
For fold1 late stage, MAE drops from 21.2180 (head-only) to 5.7781 (delta-blend correction). Mean residual moves from strong negative bias (-20.4768) toward substantially reduced underestimation (-5.7781), supporting the claim that the dominant structural error is late-stage underestimation of future increments.

## 5. Discussion
Our evidence supports a role-separated interpretation:
- TimerXL captures backbone trend information.
- TMA improves representation comparability under temporal-scale variation.
- KNN-DRR performs the main correction of future-increment residual bias.

This separation is important for scientific clarity. The observed gain should not be interpreted as backbone replacement. Instead, retrieval-based residual correction is most effective after a reasonable trend predictor is in place.

The train-only inner LOSO protocol also matters. It preserves cross-condition validation pressure without using outer test folds for tuning, improving both credibility and robustness of reported results.

## 6. Limitations
This study has clear boundaries.

First, formal validation is currently on PHM2010 only. The claims are therefore limited to this dataset and protocol rather than universal industrial forecasting.

Second, the current setting is history-aware wear forecasting. Extension to strictly sensor-only blind prediction would require an additional wear-state estimation stage.

Third, retrieval correction still depends on coverage of relevant historical patterns in the train-only library; extremely sparse late-stage coverage can weaken correction quality.

## 7. Conclusion
We presented a clean-protocol pipeline for few-shot cross-condition tool wear prediction with explicit module hierarchy: TimerXL backbone, TMA auxiliary train-time enhancement, and KNN-DRR primary residual correction. Under three-fold LOCO on PHM2010, average MAE improves from 8.7711 to 3.1784, and hardest-fold late-stage MAE is reduced from 21.2180 to 5.7781. These results indicate that the key bottleneck is structural late-stage underestimation under temporal-scale mismatch, and that train-time comparability enhancement plus retrieval-based residual correction is an effective and protocol-consistent solution.

## Appendix Note for Submission Preparation
This draft intentionally omits unverified bibliography entries to avoid citation hallucination. Add references only after programmatic verification.
