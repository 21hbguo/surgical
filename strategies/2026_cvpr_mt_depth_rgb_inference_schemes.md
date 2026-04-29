# 2026 CVPR 向 Mean Teacher + Depth 训练辅助方案
设计原则：所有方案默认推理阶段只使用 RGB；depth 作为训练期 privileged modality 提供几何、边界、遮挡和伪标签质量约束；主框架优先沿用 Mean Teacher 半监督范式，并面向腹腔镜手术器械分割中的细长结构、反光、烟雾、遮挡、组织粘连和跨术式域偏移做适配。

| 编号 | 方案名 | 主要创新点 | 训练期模态 | 推理期模态 | 临床适配点 | 实现难度 | 预期收益 |
|---|---|---|---|---|---|---|---|
| 1 | DG-CoTeach-MT | 构建 RGB student 与 depth-privileged auxiliary student 双学生互教，EMA teacher 只聚合 RGB student，depth 分支只在训练期输出高置信伪标签、边界提示和困难区域权重，最终部署丢弃 depth 分支 | RGB+depth1/depth3 | RGB | 器械轴细长、尖端小目标、组织接触边界由 depth 分支补充几何稳定性 | 中 | 在不增加推理成本的前提下提升少标注稳定性和边界召回 |
| 2 | Depth-Privileged EMA | EMA teacher 内部加入 depth privileged branch，伪标签由 RGB logits 与 depth geometry logits 融合生成，student 只接收 RGB，训练结束只保存 RGB student | RGB+depth | RGB | 反光和低对比区域中 RGB teacher 易错，depth branch 提供结构先验过滤伪标签 | 中 | 提升伪标签质量，降低 Mean Teacher confirmation bias |
| 3 | Instrument Boundary Depth MT | 从 depth 梯度、深度不连续和局部平面变化构造 boundary confidence map，对 student/teacher 边界 logits、signed distance 或 contour probability 加一致性约束 | RGB+depth | RGB | 器械边缘细、断裂多、与组织颜色接近，边界比区域 Dice 更关键 | 中 | 改善边界 Hausdorff、NSD 和器械尖端完整性 |
| 4 | Smoke-Reflection Robust MT | 训练期检测 RGB 高亮反光、烟雾模糊和低纹理区域，用 depth 稳定性决定伪标签采信权重；不可靠 RGB 区域降低一致性损失或改用 depth teacher 蒸馏 | RGB+depth | RGB | 腹腔镜常见高光、烟雾、运动模糊导致伪标签漂移 | 中 | 降低噪声伪标签对 student 的负迁移，提升复杂术野鲁棒性 |
| 5 | Occlusion-Aware Cross-Modal MT | 用 depth discontinuity 定位遮挡边界和器械-组织接触区，在这些区域引入局部 teacher agreement、局部 CutMix 和遮挡恢复损失 | RGB+depth | RGB | 器械被组织、烟雾、血液或另一把器械遮挡时，普通一致性会过度平滑 | 高 | 提升遮挡器械尖端和交叠区域召回，减少边界粘连 |
| 6 | Prototype Depth-Disentangled MT | 将原型拆成 RGB appearance prototype 与 depth geometry prototype，训练时用解耦损失减少外观和几何冗余，再把 geometry prototype 蒸馏到 RGB feature space | RGB+depth | RGB | 器械类别外观变化小但姿态、深度和接触关系变化大，几何原型更稳定 | 高 | 增强跨病人、跨器械姿态和跨术式泛化能力 |
| 7 | Ambiguity-Aware Surgical MT | 借鉴 AmbiSSL 思路，用多个轻量解码器建模标注歧义，teacher 输出伪标签分布而非单一 hard mask，边界模糊区使用 ambiguity-aware consistency | RGB+depth 可选 | RGB | 器械边界、烟雾遮挡和组织接触处存在专家标注差异，单一伪标签过硬 | 中 | 减少边界过拟合，提升不确定区域校准和泛化 |
| 8 | Depth-Guided Patch Curriculum MT | 基于 depth 边界复杂度、器械细长度、反光强度和 teacher entropy 生成 patch 难度课程，训练早期学稳定区域，后期强化尖端、边界和遮挡 patch | RGB+depth | RGB | 临床图像难度高度不均衡，随机 crop 容易浪费在背景或简单器械轴区域 | 低-中 | 提升训练效率和困难区域性能，适合快速落地到现有框架 |
| 9 | Temporal Depth-Privileged MT | 若有视频序列，训练期加入 depth-guided temporal consistency，用 depth 变化约束相邻帧伪标签传播，部署支持单帧 RGB 或 RGB 视频 | RGB video+depth | RGB | 腹腔镜视频中器械连续运动，单帧 teacher 在模糊帧上不稳定 | 高 | 提升视频连续性，减少闪烁和短时漏检 |
| 10 | Foundation-Prior Depth MT | 用 SAM/MedSAM 类基础模型产生候选 mask，depth teacher 过滤不符合器械几何的候选区域，Mean Teacher 只学习经过筛选的高质量伪标签 | RGB+depth+foundation prior | RGB | 少标注场景下基础模型容易把反光组织或器械影子误分，depth 可做结构校验 | 高 | 提升极少标注启动质量，增强方法新颖性和可解释性 |

## 优先级建议
首推 `DG-CoTeach-MT`、`Prototype Depth-Disentangled MT`、`Depth-Guided Patch Curriculum MT`。`DG-CoTeach-MT` 最贴近现有 `mt_depth_guider_proto_*` 路线，训练期 depth、推理期 RGB 的设定清晰；`Prototype Depth-Disentangled MT` 创新强度最高，适合作为 2026 CVPR 主方案；`Depth-Guided Patch Curriculum MT` 改动成本最低，适合作为稳定增益模块和消融组件。

## 参考方向
- CVPR 2025 Boost the Inference with Co-training: A Depth-guided Mutual Learning Framework for Semi-supervised Medical Polyp Segmentation
- CVPR 2025 Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation
