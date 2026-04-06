# AEI 写作参考对齐说明

本项目将 AEI 论文《Robust tool wear prediction under novel operating conditions via physics-guided unsupervised domain adaptation》仅作为写作参考，而不作为方法路径参考。可借鉴的部分主要包括以下四点：其一，引言从制造重要性、刀具磨损对质量与停机的影响、再到真实工业场景中的部署困难，形成了由应用价值到技术问题的递进开场；其二，相关工作按“问题类型与方法家族”而非按零散文献堆砌进行归类，便于突出本文所解决的是 few-shot cross-condition 预测中的结构性误差，而不是一般性的建模精度问题；其三，实验协议部分对训练、验证和测试边界的表述清晰克制，适合作为本文 zero-leakage、inner LOSO 选参与 outer test 单次评估写法的风格参照；其四，贡献表述具有分层性，能够把平台、辅助机制和核心创新区分开来，这与本文固定的方法层级完全一致。

AEI 论文中以下内容不能借入本文。首先，不能把本文改写为无监督域适应故事，也不能使用 target-domain adaptation、pseudo-label transfer、physics-guided calibration 等作为主方法框架。其次，不能把 source/target 域对齐当作本文的中心问题，因为本文的核心矛盾不是“无标签目标域适配”，而是 few-shot cross-condition 情况下由磨损速率差异引起的时间尺度错位，以及由此诱发的 late-stage future-increment underestimation。再次，不能把 AEI 中“通过域间分布对齐实现鲁棒预测”的论证逻辑移植为本文主线，因为本文的主创新是 KNN-DRR 的残差检索修正，而不是域对齐本身。

因此，本文与 AEI 的根本区别应写清楚：AEI 解决的是 novel operating conditions 下的无监督域适应预测问题，而本文解决的是 PHM2010 上的 few-shot cross-condition 刀具磨损预测问题。本文的方法层级固定为 TimerXL 提供基础时序表示，TMA 作为训练期的时间尺度增强以提升跨速率可比性，KNN-DRR 作为主创新模块对未来磨损增量残差进行修正。正式结果严格建立在 zero-leakage 协议、train-only inner LOSO 选参与 outer test 单次评估的证据链之上。
