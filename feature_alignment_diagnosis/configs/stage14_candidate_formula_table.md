# 阶段十四-候选特征公式表（Step 1）

更新时间：2026-03-17  
状态：仅定义候选与优先级，不执行训练

## 1. 当前可直接使用的 `td28` 特征定义

`td28` 来自 7 个通道 × 4 个时域统计量，按通道顺序展开：

1. `Feat_{4*(ch-1)+1}`：`mean(x_ch)`  
2. `Feat_{4*(ch-1)+2}`：`std(x_ch)`  
3. `Feat_{4*(ch-1)+3}`：`rms(x_ch) = sqrt(mean(x_ch^2))`  
4. `Feat_{4*(ch-1)+4}`：`range(x_ch) = max(x_ch) - min(x_ch)`

其中 `ch=1..7`。

## 2. 与当前主线的对应关系

1. `RMS7` 基线特征：`Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27`。  
2. 当前 `RMS7 + Feat_4`：在 `RMS7` 基础上增加 `Feat_4`（通道1的 `range`）。

## 3. 阶段十四“单通道 +1”候选池（第一批）

说明：以下候选均来自已提取的 `td28`，可直接做 `+1`；候选选择遵循“先单通道，再多通道”。

1. `Feat_8`：通道2 `range`
2. `Feat_10`：通道3 `std`
3. `Feat_12`：通道3 `range`
4. `Feat_16`：通道4 `range`
5. `Feat_18`：通道5 `std`
6. `Feat_20`：通道5 `range`
7. `Feat_22`：通道6 `std`
8. `Feat_24`：通道6 `range`
9. `Feat_28`：通道7 `range`

## 4. 顶刊常见扩展候选（第二批，待实现）

当前仅定义公式，后续实现后再进入单通道 `+1` 测试队列。

1. 形状类：`kurtosis`、`skewness`、`crest_factor`、`impulse_factor`、`shape_factor`
2. 频域类：`band_energy_ratio`、`spectral_centroid`、`main_peak_freq`
3. 时频类：`STFT` 高频能量占比、`WPT` 节点能量占比

## 5. 说明

1. 本文件仅用于阶段十四的候选定义与追踪。  
2. 所有候选准入仍遵循 `实施计划2.md` 第十四节硬门槛与 `hardest fold` 规则。  
3. 本文件不参与任何模型拟合，不触发数据泄露风险。
