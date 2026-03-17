# 模块规划

## 计划模块

1. `domain_split.py`
   - 定义每一折的域划分。
   - 区分“验证目标域”与“盲测目标域”。

2. `scaler_fit.py`
   - 仅在训练域上拟合标准化器。
   - 统一应用到 train/val/test。

3. `wear_binning.py`
   - 仅用训练域标签拟合磨损分位数分段阈值。
   - 将固定分段边界应用到 val/test 进行着色。

4. `embedding_plot.py`
   - PCA 二维图（主图）：按域着色、按磨损分段着色。
   - UMAP 二维图（辅图）：可选补充图。

5. `distribution_plot.py`
   - 关键特征跨域分布图（KDE/箱线图）。

6. `alignment_metrics.py`
   - MMD
   - CORAL 距离
   - 域分类 AUC（单特征）
   - 域分类 AUC（整体特征集）

7. `report_writer.py`
   - 保存指标表和图像索引。
   - 输出每折零泄露检查记录。
