# novelty 风险审计

时间基准：`2026-06-24`

## 结论

当前这条线能做，但不能再写成：

- `depth helps RGB SSL`
- `use depth to improve semi-supervised surgical segmentation`
- `uncertainty + depth fusion`

按 2026 的文献状态，这些表述都不够新，且会直接撞上 `RDNet (CVPR 2025)`、`SegMatch (Scientific Reports 2025)`、`Endo-SemiS (MIDL 2026)` 以及一整条 pseudo-label reliability 文献线。

能成立的表述必须收缩到：

> 在 RGB-only 半监督手术器械分割中，foundation-model 生成的 pseudo-depth 既可能提供局部结构证据，也可能把错误几何注入伪标签学习；因此需要对 depth-derived geometry 做显式仲裁，只在通过验证的区域使用其调节未标注监督。

## 高风险撞车点

### 1. 与 RDNet 的主线重叠最高

`RDNet` 已经覆盖：

- 训练期使用 depth 辅助
- 推理期只用 RGB
- depth 分支指导主分支学习
- 利用 depth 找 difficult regions

如果你的叙事是：

- `train-depth test-RGB`
- `depth branch improves RGB SSL`
- `depth-aware hard patch supervision`

那么会被直接归入 RDNet 同类改造。

必须拉开的点：

- 你不是让 depth 分支直接改语义伪标签
- 你研究的是 `noisy pseudo-depth 何时不该被信任`
- depth 只参与 `boundary / continuity / supervision weighting / arbitration`
- 重点是 `harm suppression`，不是 `benefit amplification`

### 2. 与 U2PL / CorrMatch / When Confidence Fails 的可靠性主线重叠很高

这些工作已经把以下命题做得很充分：

- 低质量伪标签不能简单丢弃
- confidence / entropy 不是完美的选择标准
- 难像素需要差异化利用

如果你最后只是：

- `entropy + depth gradient`
- `confidence gating + consistency`

审稿人会认为这是把现有可靠性筛选再加一个 depth cue。

必须拉开的点：

- 你的可靠性不是 semantic confidence 本身
- 你的核心对象是 `depth-derived geometry evidence`
- 你的仲裁单元是 `结构区域`
- 你的主证据是 `geometry-semantic conflict / counterfactual stability / temporal stability`

### 3. 与 DFormer / DFormerv2 / PrimKD 的 RGB-D 叙事容易混淆

这些工作已经证明：

- depth 可以被当作 geometry prior
- RGB 仍应保持主模态地位
- depth 不一定需要完全对称编码

如果你表述不严，审稿人会把你看成：

- 一个半监督版 RGB-D segmentation
- 一个 depth-guided encoder + consistency regularization 组合

必须明确：

- 你不是测试期依赖 depth 的 RGB-D 模型
- 你不是研究更好的 RGB-D fusion
- 你是 `RGB-only deployment` + `privileged pseudo-geometry during training`

### 4. 2026 的 Endo-SemiS 提高了手术视频方向门槛

`Endo-SemiS (MIDL 2026)` 已经把：

- 手术/内镜视频
- 伪标签可靠性
- 双网络监督
- 时序纠错

放进同一条 robust SSL 主线里。

这带来的直接结果是：

- 你如果完全不谈视频特性，手术场景特异性会偏弱
- 你如果谈时序，只谈“相邻帧更平滑”，也不够

建议处理：

- 时序作为增强项，而不是第一主贡献
- 只把它用于 `geometry validation`，不扩成一整套复杂视频框架
- 论文里明确写：时序的意义在于过滤 pseudo-depth jitter，不是另起一条视频学习主线

### 5. foundation-model pseudo-label refinement 已经变成显性竞争线

2025-2026 的 `SemiSAM+`、`SSL-MedSAM2`、foundation-model guided iterative prompting / pseudo-labeling 说明：

- foundation model 已经被大量用于生成和修正伪标签
- “借助外部大模型改善半监督分割”本身不新

这意味着你的差异点不能写成：

- `利用 foundation model 增强半监督分割`
- `引入额外先验提升 pseudo-label 质量`

必须明确：

- 这些工作主要利用 foundation model 生成 `语义伪标签`
- 你的工作利用 foundation model 产生 `noisy pseudo-depth geometry`
- 你的核心难点不是语义伪标签 refinement，而是 `几何证据的选择性蒸馏与抑制`

### 6. unreliable modality fusion 文献已足以否掉泛化表述

`conflict-guided evidential fusion`、`reliability-aware fusion under sensor degradation` 这类工作已经把：

- 模态冲突
- 证据折扣
- 退化传感器可靠融合

做成成熟问题。

因此不能把你的方法写成：

- `reliability-aware multimodal fusion`
- `unreliable depth fusion for segmentation`

否则会被归到“测试期多模态可靠融合”的已知路线。

必须坚持的边界是：

- 测试期只有 RGB
- pseudo-depth 只在训练期出现
- 目标是防止错误几何放大半监督 confirmation bias，而不是做更稳的推理期多模态融合

## 当前方案最容易被否掉的说法

以下说法不建议再用：

1. `深度信息提升半监督分割性能`
2. `几何信息帮助伪标签生成`
3. `深度可信度建模`
4. `RGB-D 半监督手术器械分割`
5. `用 monocular depth 增强 segmentation`

原因：

- 1 和 2 太宽，已被覆盖
- 3 太像普通 uncertainty/reliability heatmap，缺乏辨识度
- 4 会把问题导向测试期依赖 depth
- 5 像工程增强，不像期刊问题

## 仍然可成立的主张

### 推荐主张

> monocular pseudo-depth in surgical scenes should not be treated as uniformly beneficial supervision; it must be selectively validated before affecting unlabeled structural learning.

中文压缩版：

> 手术场景中的单目伪深度不能被默认当作稳定有益的监督信号，必须先做局部几何验证，再参与未标注结构监督。

### 推荐关键词

- `geometry arbitration`
- `privileged pseudo-geometry`
- `structural supervision selection`
- `counterfactual geometry evidence`
- `temporal geometry validation`

不建议继续主打：

- `trust map`
- `reliability map`

因为这两个词太容易被理解成普通 uncertainty calibration。

## 最稳的主线收敛

### 主线

`RGB-only Semi-supervised Surgical Instrument Segmentation with Geometry Arbitration from Pseudo-Depth`

核心定义：

- semantic class 决定权来自 RGB teacher
- pseudo-depth 不改类别，只改结构监督强度
- 几何信息只用于：
  - hard pseudo-label weight
  - boundary consistency
  - continuity / perturbation consistency
  - ignore / soften unreliable regions

### 备选加强项

如果结果允许，再加：

- counterfactual depth perturbation stability
- temporal consistency of depth boundary

但它们只能服务 `geometry validation`，不要展开成第二主贡献。

## 必须补的证据缺口

当前想把论文讲稳，至少需要补以下证据：

1. `high-trust vs low-trust` 区域的伪标签准确率差异
2. `Q_geo` 或仲裁权重对 teacher error 的 AUROC / AUPRC
3. clean depth 与 corrupted depth 下的性能变化
4. `RGB-only`、`RGB+depth concat`、`DepthGuiderV4`、`RDNet`、`proposed` 的并列表
5. 边界区、细长器械区、depth discontinuity band 的局部指标

缺任一项，`geometry arbitration` 都容易退化成叙事概念。

## 最终判断

这条线在 2026 仍可做，但可做的前提不是“再造一个更强 depth 模块”，而是把问题从：

`depth can help`

改成：

`pseudo-depth can help or harm, so its geometric evidence must be audited before it modulates unlabeled structural supervision`

一句话总结：

`你要证明的不是 depth 有用，而是 depth 什么时候不该被用，以及你如何阻断它污染半监督伪标签学习。`
