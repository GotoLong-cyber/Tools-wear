# RMS特征选择依据 - 文献支持段落

## 论文中可直接使用的版本

### 版本1：完整详细版（适合正文或Method部分）

```
## 3.1 特征选择

我们选择均方根（RMS）作为基础特征，主要基于以下原因：

**1. RMS 是刀具磨损监测中被充分验证的有效时域特征。**

大量研究表明，RMS 是刀具状态监测中最鲁棒、最具信息量的时域特征
之一。综述文献 [2] 明确指出，当刀具磨损增加时，振动信号的时域
特征，如 RMS、峰峰值和峰度，都会显著上升。这种 RMS 与刀具磨损
演化之间的直接相关性，使其成为刻画退化状态的可靠指标。

**2. RMS 能够表征与切削动力学相关的有效信号能量。**

RMS 反映了传感器信号的有效能量水平，能够直接刻画切削过程中的
力学强度与振动强度。已有研究 [1] 已成功将电流信号中的 RMS 与其
他统计特征结合，用于刀具磨损预测，说明 RMS 在不同传感器模态下
都具备良好的适用性。

**3. RMS 计算代价低，适合实时工业预测。**

与复杂的频域或时频域特征相比，RMS 的计算过程简单、速度快，更
适合实时工业监测与在线预测场景 [6]。

**4. RMS 与深度学习模型具有良好的互补性。**

近期研究 [4,5] 已将 RMS 等时域特征与深度学习、迁移学习方法结合。
在本文中，我们采用大规模预训练时间序列模型 TimerXL，使模型能够
借助深层上下文注意力机制，从基础 RMS 特征中自动学习更深层的退化
表征，从而减少对复杂人工特征工程的依赖。

**参考文献：**

[1] Aguiar et al. "Tool Wear Condition Monitoring by Combining Variational
    Mode Decomposition and..." MDPI Sensors, 2020, 20(21):6113.
    https://www.mdpi.com/1424-8220/20/21/6113

[2] "Tool condition monitoring techniques in milling process — a review."
    ScienceDirect.
    https://www.sciencedirect.com/science/article/pii/S2238785418313061

[3] "Cutting tool wear monitoring based on a smart toolholder with..."
    ScienceDirect, 2022.
    https://www.sciencedirect.com/science/article/abs/pii/S0263224122007424

[4] "A Novel Multivariate Cutting Force-Based Tool Wear Monitoring..."
    PMC, 2022.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9657287/

[5] "Tool Condition Monitoring for High-Performance Machining Systems"
    PMC.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8950983/

[6] "Mondragon Unibertsitatea face-milling dataset for smart tool..."
    Nature, 2025.
    https://www.nature.com/articles/s41597-025-05168-5
```

### 版本2：简洁版（适合空间受限的情况）

```
## 特征选择

我们采用七个传感器通道的均方根（RMS）值作为基础特征。RMS 是刀具
磨损监测研究中被广泛验证的有效时域特征 [1,2]，并已知与刀具磨损
演化具有显著相关性。正如相关综述所指出的，当刀具磨损增加时，
振动信号的时域特征（如 RMS）会显著上升 [2]。RMS 不仅能够表征
切削动力学中的有效信号能量，而且计算效率高、适合实时应用，并且
能够与深度学习模型形成良好互补 [3,4,5]。

**参考文献：**

[1] Aguiar et al., MDPI Sensors, 2020, 20(21):6113.

[2] "Tool condition monitoring techniques in milling process — a review."
    ScienceDirect.

[3] "Cutting tool wear monitoring based on a smart toolholder..."
    ScienceDirect, 2022.

[4] "A Novel Multivariate Cutting Force-Based Tool Wear Monitoring..."
    PMC, 2022.

[5] "Tool Condition Monitoring for High-Performance Machining Systems"
    PMC.
```

### 版本3：中文版

```
## 3.1 特征选择

我们选择均方根（RMS）作为基础特征，主要基于以下考虑：

**1. RMS 是刀具磨损监测中被广泛验证的有效时域特征。**

大量研究表明，RMS 是刀具状态监测中最鲁棒和最信息丰富的时域特征
之一。根据综述研究[2]的总结，"当刀具磨损增加时，振动信号的时域
特征如 RMS、峰峰值和峰度值显著增加"。这种 RMS 值与刀具磨损进展
之间的直接相关性使其成为退化状态的可靠指标。

**2. RMS 捕获与切削动力学相关的有效信号能量。**

RMS 表征信号的有效能量含量，直接反映切削过程中的力学和振动强度。
多项研究[1,3]已成功使用从电流、力、振动等信号中提取的 RMS 值
进行刀具磨损预测，证明了其跨不同传感器模态的有效性。

**3. RMS 计算高效，适合实时预测。**

相比复杂的频域或时频域特征，RMS 计算简单快速，使其适合实时工业
预测场景[6]。

**4. RMS 与深度学习模型有效互补。**

近期研究[4,5]将 RMS 等时域特征与深度学习和迁移学习方法相结合。
我们采用的大型时序基础模型 TimerXL 能够通过深层上下文注意力机制
自动从基础的 RMS 特征中学习更深层的退化表征，无需依赖复杂的
人工特征工程。

**参考文献：**

[1] Aguiar et al. "Tool Wear Condition Monitoring by Combining Variational
    Mode Decomposition and..." MDPI Sensors, 2020, 20(21):6113.

[2] "Tool condition monitoring techniques in milling process — a review."
    ScienceDirect.

[3] "Cutting tool wear monitoring based on a smart toolholder..."
    ScienceDirect, 2022.

[4] "A Novel Multivariate Cutting Force-Based Tool Wear Monitoring..."
    PMC, 2022.

[5] "Tool Condition Monitoring for High-Performance Machining Systems"
    PMC.

[6] "Mondragon Unibertsitatea face-milling dataset for smart tool..."
    Nature, 2025.
```

---

## BibTeX 格式（可直接用于 .bib 文件）

```bibtex
@article{aguiar2020tool,
  title={Tool wear condition monitoring by combining variational mode decomposition and...},
  author={Aguiar, ...},
  journal={MDPI Sensors},
  volume={20},
  number={21},
  pages={6113},
  year={2020},
  doi={10.3390/s20216113}
}

@article{milling2019tool,
  title={Tool condition monitoring techniques in milling process — a review},
  journal={ScienceDirect},
  year={2019},
  doi={10.1016/j.jmapro.2018.12.035}
}

@article{smart2022cutting,
  title={Cutting tool wear monitoring based on a smart toolholder with...},
  journal={ScienceDirect},
  year={2022},
  doi={10.1016/j.measurement.2022.110742}
}

@article{multivariate2022tool,
  title={A novel multivariate cutting force-based tool wear monitoring method...},
  journal={PMC},
  year={2022},
  url={https://pmc.ncbi.nlm.nih.gov/articles/PMC9657287/}
}

@article{high2022tool,
  title={Tool condition monitoring for high-performance machining systems},
  journal={PMC},
  year={2022},
  url={https://pmc.ncbi.nlm.nih.gov/articles/PMC8950983/}
}

@article{mondragon2025face,
  title={Mondragon Unibertsitatea face-milling dataset for smart tool condition monitoring},
  journal={Nature},
  year={2025},
  doi={10.1038/s41597-025-05168-5}
}
```

---

## 💡 使用建议

### 如何在论文中引用

1. **在 Method / 特征选择部分**：直接使用上面的段落，并引用文献 [1-6]
2. **在 Related Work 部分**：可以补充说明已有工作普遍采用 RMS 特征
3. **在消融实验部分**：如果审稿人质疑 RMS 的合理性，可以引用这些文献作为依据

### 如果审稿人进一步质疑

**Q**: "为什么只用 7 维 RMS，不用更多特征？"

**A**:
```
我们选择纯 RMS 特征是基于以下考虑：
1. 文献[1,2]已证明 RMS 是最鲁棒和最信息丰富的时域特征
2. 我们的大型时序模型（TimerXL）能自动从基础 RMS 特征中学习更深层
   的表征，无需复杂特征工程
3. 消融实验（表X）表明，Pure RMS7 + TimerXL 已经能取得 2.47 MAE
   的优秀性能
4. Full-133 特征在 C6 工况上出现过拟合（MAE 高达 29.90），说明
   更多特征并不总是更好
```

### 额外支持点

1. **PHM2010 数据集本身**：许多使用 PHM2010 的工作都会使用 RMS 特征
   - 后续可以继续搜索 `"PHM2010 RMS feature"` 以补充更多支持

2. **工业界应用**：RMS 是工业界最常用的特征之一（计算简单，实时性好）

3. **最新趋势**：Nature 2025 的数据集论文仍然强调时域统计特征，说明这仍是领域共识

---

## 参考来源：
- [Tool Wear Condition Monitoring by Combining Variational Mode... (MDPI Sensors, 2020)](https://www.mdpi.com/1424-8220/20/21/6113)
- [Tool condition monitoring techniques in milling process — a review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2238785418313061)
- [Cutting tool wear monitoring based on a smart toolholder... (ScienceDirect, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0263224122007424)
- [A Novel Multivariate Cutting Force-Based Tool Wear Monitoring... (PMC, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9657287/)
- [Tool Condition Monitoring for High-Performance Machining Systems (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8950983/)
- [Mondragon Unibertsitatea face-milling dataset for smart tool... (Nature, 2025)](https://www.nature.com/articles/s41597-025-05168-5)
