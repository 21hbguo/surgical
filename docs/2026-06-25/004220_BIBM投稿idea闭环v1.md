# BIBM 2026 投稿 idea 与闭环方案

版本：v1.0
日期：2026-06-25
目标会议：IEEE BIBM 2026
投稿约束：Full paper 截止 2026-07-05，8 页 IEEE 双栏，双盲

## 1. 最终推荐 idea

推荐主线：

**Geometry-conditioned pseudo-label risk allocation for semi-supervised surgical instrument segmentation**

中文表述：

**面向半监督手术器械分割的几何条件伪标签风险分配**

核心句：

> 单目伪深度不直接修正语义伪标签，而是作为结构风险先验，决定未标注像素应接受 hard pseudo-label、soft consistency，还是 boundary consistency。

这条线比“depth helps SSL”更具体，比“depth trust map”更稳，比“temporal-causal geometry arbitration”更适合 BIBM 8 页和当前 10 天剩余时间。

## 2. Problem anchor

底层问题：

半监督手术器械分割中，未标注数据的伪标签错误不是均匀分布的，而是集中在细器械边界、遮挡、反光、器械-组织接触和局部深度突变区域。

必须解决的瓶颈：

现有 SSL 方法主要依赖 confidence、entropy、weak-to-strong augmentation 或互教机制判断伪标签是否可靠，但这些信号不能充分刻画手术场景中的结构性边界风险。直接加入 pseudo-depth 又默认深度总是有益，会把错误几何注入分割训练。

非目标：

- 不做 RGB-only 推理主张，因为当前训练和测试仍输入 depth。
- 不声称预测 calibrated depth trust map，因为没有 depth trust ground truth。
- 不做时序因果方案，因为 10 天内无法闭合实验。
- 不主推 GAN、prototype、text 或多模块堆叠。

成功条件：

论文能清楚证明：pseudo-depth 的价值不是直接提供语义标签，而是帮助定位未标注监督中的结构风险，并通过不同监督形式降低边界伪标签错误。

## 3. 核心 thesis

一句话 thesis：

> Monocular pseudo-depth is useful not as a semantic label corrector, but as a geometry-conditioned risk prior that reallocates unlabeled supervision in semi-supervised surgical instrument segmentation.

对应中文：

> 单目伪深度的价值不在于替代 RGB teacher 修正类别标签，而在于提供几何条件风险先验，用来重新分配未标注像素的监督方式。

## 4. 方法闭环

### 4.1 输入与基础框架

基础框架使用 Mean Teacher。

输入包括：

- RGB image
- 预计算 monocular pseudo-depth
- labeled mask
- unlabeled image-depth pair

Teacher 产生语义 pseudo-label 和 confidence。Pseudo-depth 只参与 risk prior 计算和 depth-guided representation，不直接决定语义类别。

### 4.2 风险定位

风险信号由两类核心量组成：

- Teacher uncertainty：表示语义预测不确定区域。
- Depth discontinuity：表示几何结构变化区域。

推荐 BIBM 最小版本使用：

```text
R = norm(U_t) * norm(G_d)
```

其中 `U_t` 是 teacher entropy，`G_d` 是局部归一化 depth 的 Sobel gradient magnitude。

不建议把 conflict term 作为主贡献，因为当前消融显示完整默认版本不是最优，且会让方法显得更手工。

### 4.3 监督分配

根据 teacher confidence 和 geometry-conditioned risk，把未标注像素分为两类：

- 低风险高置信区域：使用 hard pseudo-label CE。
- 高风险结构区域：避免 hard pseudo-label，使用 soft consistency 和 boundary consistency。

关键原则：

**Depth 不改语义标签，只改变监督方式和监督强度。**

### 4.4 DepthGuiderV4 的位置

DepthGuiderV4 作为支持模块保留，不作为唯一创新点。

它的作用是证明 pseudo-depth 作为 feature source 有稳定收益；risk allocation 的作用是证明 pseudo-depth 还能作为 supervision routing prior。

论文中应写成：

- DepthGuiderV4 provides geometry-aware representation.
- GeoRisk allocation provides geometry-conditioned unlabeled supervision.

不要写成两个平行大贡献。

## 5. 论文闭环结构

### 5.1 Introduction 闭环

手术器械 SSL 难点：

伪标签错误集中在结构区域。

现有方法不足：

confidence/entropy 能反映语义不确定性，但不能单独解释边界、遮挡和深度突变处的结构风险。

Pseudo-depth 机会：

单目深度能提供局部几何结构，如 depth discontinuity 和 instrument-tissue interface。

Pseudo-depth 风险：

手术场景中的单目深度不稳定，不能直接当作语义监督或第二模态无脑融合。

本文核心：

用 pseudo-depth 做 risk prior，而不是 semantic corrector。

### 5.2 Method 闭环

方法只回答一个问题：

**未标注像素应该接受哪种监督？**

对应机制：

- Teacher 给语义 pseudo-label。
- Depth + uncertainty 给 risk prior。
- Risk prior 决定 hard label 或 soft/boundary consistency。
- DepthGuiderV4 提供几何特征增强。

### 5.3 Experiment 闭环

每个实验必须对应一个 claim：

| Claim | 实验 | 证明内容 |
|---|---|---|
| 方法提升半监督分割 | Task1 5/10/20/40% 主表 | 整体性能有效 |
| 几何结构区域确实受益 | BF1、HD95、ASSD | 改善边界而不只是 Dice |
| depth+uncertainty 是有效 risk prior | risk source ablation | 几何风险信号有用 |
| risk 分区能区分伪标签质量 | low-risk vs high-risk pseudo-label accuracy | 分区不是任意 mask |
| 方法不是简单 depth fusion | MT vs MT-DGv4 vs GeoRisk-DGv4 | risk allocation 与 feature injection 区分 |

## 6. 建议标题

首选：

**GeoRisk-SPC: Geometry-Conditioned Pseudo-label Risk Allocation for Semi-supervised Surgical Instrument Segmentation**

备选：

**Geometry-Conditioned Unlabeled Supervision for Semi-supervised Surgical Instrument Segmentation**

不建议：

- TrustDepth
- Temporal-Causal Geometry Arbitration
- RGB-only Depth Distillation
- Reliability-aware RGB-D Surgical Segmentation

## 7. 建议贡献写法

贡献 1：

We identify geometry-structured pseudo-label risk in semi-supervised surgical instrument segmentation, where errors concentrate around thin boundaries, occlusion interfaces, and depth discontinuities.

贡献 2：

We propose a geometry-conditioned risk allocation strategy that uses monocular pseudo-depth and teacher uncertainty to route unlabeled pixels to hard pseudo-label, soft consistency, or boundary consistency supervision.

贡献 3：

We integrate the risk allocation with a depth-guided encoder and validate the framework using main results, boundary-aware metrics, risk-mask audits, and risk-source ablations.

## 8. 当前证据支撑情况

已能支撑：

- GeoRisk-SPC-DGv4 在 Task1 5/10/20/40% 有完整结果。
- 20% 下 boundary metrics 明显优于 MT、UAMT、SegMatch。
- risk audit 显示 low-risk 区域伪标签准确率明显高于 outside-low-risk 区域。
- risk source ablation 显示 depth+uncertainty 是强配置。

需要注意：

- 默认 full risk map 不是最优，不能强行写成所有组件单调有效。
- `risk_source=depth_uncertainty` 如果作为最终方法，需要统一主表和方法描述。
- 当前不是 RGB-only inference，不能写训练用 depth、测试不用 depth。
- 4-fold 统计无法给出严格显著性，不能夸大统计结论。

## 9. 最小补强实验

如果还有 GPU，优先级如下：

1. 补 `risk_source=depth_uncertainty` 在 5/10/40% 的全 fold，若结果稳定，作为最终方法。
2. 统一 `GeoRisk-SPC-DGv4`、`Ablation_depth_uncertainty`、boundary metrics 的数字来源。
3. 生成 2-3 张真实 risk map 可视化，展示 RGB、depth、uncertainty、risk、pseudo-label error、prediction。
4. 若时间不足，不补新方法，只整理已有 20% 消融和 risk audit。

不建议补：

- 时序实验。
- depth corruption curve。
- RGB-only inference。
- GAN 消融。
- Task3 深入分析。

这些会打开新问题，无法在 BIBM 截止前闭合。

## 10. BIBM 投稿版本推荐叙事

最终 BIBM 版摘要逻辑：

1. 半监督手术器械分割受伪标签错误影响，错误集中在结构区域。
2. Confidence-only pseudo-label selection 难以捕捉几何结构风险。
3. Monocular pseudo-depth 提供局部几何线索，但不应直接修正语义伪标签。
4. 提出 GeoRisk-SPC，用 depth discontinuity 和 teacher uncertainty 估计 risk prior。
5. Risk prior 将未标注像素分配到 hard pseudo-label、soft consistency、boundary consistency。
6. 在 EndoVis 2017 上验证 Dice、Boundary F1、HD95、ASSD、risk audit 和 ablation。

## 11. 需要避免的写法

禁止主张：

- “我们解决 RGB-only 推理问题。”
- “我们提出 calibrated depth trust estimation。”
- “我们证明 pseudo-depth always helps。”
- “我们做 temporal-causal geometry validation。”
- “我们的方法是通用 RGB-D reliability fusion。”

推荐主张：

- “pseudo-depth is used as a geometry-conditioned risk prior.”
- “depth does not replace semantic pseudo-labels.”
- “risk allocation changes the form of unlabeled supervision.”
- “boundary metrics and pseudo-label audit support the structural-risk hypothesis.”

## 12. 最终判断

今年 BIBM 最小且最闭合的 idea 是：

**用单目伪深度提供几何结构风险，而不是语义纠错；再用该风险分配未标注监督形式。**

这条线的优点：

- 问题具体。
- 方法小。
- 与已有代码一致。
- 现有实验能支撑。
- 不需要引入新大模块。
- 适合 BIBM 8 页和 2026-07-05 截止时间。

下一步不是继续扩展 idea，而是把论文、表格、图和文字全部压到这个唯一闭环上。
