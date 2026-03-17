# 特征提取与树选择（阶段C）

本目录用于“基于原始数据重新提取特征 + 树模型特征选择”，不依赖已有 `full133 npz` 结果。

## 目录文件

- `pipeline_tree_selection.py`
  - 从 `dataset/c1,c4,c6` 原始 `c_*_xxx.csv` 与 `*_wear.csv` 读取数据。
  - 先按走刀级提取基础特征（当前支持 `avg7` / `td28`）。
  - 严格只在训练域（默认 `c1,c4`）训练切片上做树选择。
  - 输出 `keep_features_tree_*.txt`、重要性表、以及可直接训练的 `selected npz`。
- `run_tree_pipeline.sh`
  - 一键运行默认配置（`td28 + top_k=16 + c1,c4->c6`）。

## 默认协议（与你当前主实验一致）

- 粒度：走刀级（pass-level）
- 训练域：`c1,c4`
- 测试域：`c6`
- 划分：`split_ratio=0.8`
- 标签聚合：`wear_agg=max`
- 树选择：`RandomForestRegressor`，3 seeds 平均重要性
- 泄漏约束：仅用训练域训练切片做特征选择

## 运行

```bash
bash feature_extraction/run_tree_pipeline.sh
```

## 输出

默认输出到：`dataset/passlevel_tree_select/`

- `base_td28/`
  - `c1_passlevel_td28.npz`
  - `c4_passlevel_td28.npz`
  - `c6_passlevel_td28.npz`
- `selected_td28_k16/`
  - `keep_features_tree_td28_k16.txt`
  - `tree_importance_td28_k16.csv`
  - `selection_meta.json`
  - `c1_passlevel_td28_tree_k16.npz`
  - `c4_passlevel_td28_tree_k16.npz`
  - `c6_passlevel_td28_tree_k16.npz`

## 训练接入提示

下游训练时可直接把 `data_path` 指向新 `npz`，例如：

- `data_path=c1_passlevel_td28_tree_k16.npz`
- `root_path=.../dataset/passlevel_tree_select/selected_td28_k16`

并保持你当前 `dual-loader + 物理约束` 协议不变。
