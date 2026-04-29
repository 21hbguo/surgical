# 2026 CVPR 向 Mean Teacher + 训练期 Depth 辅助方案评估
设计原则：推理阶段只使用 RGB；depth 只作为训练期 privileged modality；主框架围绕 Mean Teacher 半监督；任务限定为腹腔镜手术器械分割，重点处理细长器械、尖端、小目标、反光、烟雾、遮挡、组织粘连、边界模糊和跨术式域偏移。以下评分按 2026 CVPR 投稿强度评估，重点看是否避开已有工作、是否有明确临床问题闭环、是否能形成可验证机制。

| 编号 | 方案名 | 主要创新点 | 已有相关工作覆盖度 | 2026 CVPR 潜力 | 风险判断 | 推荐处理 |
|---|---|---|---|---|---|---|
| 1 | DG-CoTeach-MT | RGB student 与 depth-privileged auxiliary student 双学生互教，depth 分支只训练期参与，推理丢弃 | 高 | 低-中 | CVPR 2025 已有 depth-guided mutual learning 用于半监督医学分割，核心范式高度接近；只迁移到腹腔镜器械分割不够 | 不做主方案，只作为 baseline 或对照 |
| 2 | Depth-Privileged EMA | EMA teacher 内加入 depth privileged branch，生成更可靠伪标签，student 仍为 RGB-only | 高 | 中 | privileged modality、RGB-D distillation、geometry-aware teacher 已有大量语义分割工作；单独提出不新 | 可作为 teacher 设计组件，不能单独投稿 |
| 3 | Instrument Boundary Depth MT | depth 梯度、深度不连续和局部几何变化约束器械边界一致性 | 中 | 中 | depth 边界辅助本身不新；新意来自腹腔镜器械尖端、细长轴、组织接触边界的专门建模 | 作为边界分支加入主方案 |
| 4 | Smoke-Reflection Robust MT | 将烟雾、反光、模糊建模为 teacher reliability 问题，用 depth 稳定性控制伪标签采信和一致性权重 | 中-低 | 高 | 完全做“鲁棒伪标签筛选”容易像工程技巧；必须显式建模腹腔镜退化因素并做分组验证 | 推荐作为主创新之一 |
| 5 | Occlusion-Aware Cross-Modal MT | 用 depth discontinuity 定位遮挡、器械-组织接触和器械交叠区，进行局部伪标签修正与一致性约束 | 中-低 | 高 | 若只做 depth edge mask 不够；需要把遮挡区域定义成临床结构问题，并证明 RGB-only student 学到了遮挡先验 | 推荐作为主创新之一 |
| 6 | Prototype Depth-Disentangled MT | 将 RGB appearance prototype 与 depth geometry prototype 解耦，训练期把 geometry prototype 蒸馏到 RGB feature space | 中 | 高 | RGB-D 解耦蒸馏已有相关工作；但“半监督 + depth privileged prototype + surgical instrument”仍有空间 | 推荐作为核心技术主线 |
| 7 | Ambiguity-Aware Surgical MT | 多解码器或分布式伪标签建模标注歧义，边界模糊区使用 ambiguity-aware consistency | 高 | 中-低 | CVPR 2025 AmbiSSL 已覆盖医学半监督标注歧义；没有多专家标注或新歧义基准时创新不足 | 不做主方案，除非补充多专家标注实验 |
| 8 | Depth-Guided Patch Curriculum MT | 基于 depth 边界复杂度、teacher entropy 和器械细长度选择困难 patch，做课程式半监督训练 | 高 | 低 | depth-guided patch augmentation/curriculum 已被近期 depth-guided mutual learning 覆盖较近 | 只作为辅助训练技巧或消融 |
| 9 | Temporal Depth-Privileged MT | 视频训练期用 depth-guided temporal consistency 约束相邻帧伪标签传播，推理支持 RGB-only | 中 | 中-高 | 视频时序一致性已有大量工作；新意取决于是否有稳定 depth 和遮挡恢复机制 | 有连续帧数据时可作为第二主线 |
| 10 | Foundation-Prior Depth MT | SAM/MedSAM 生成候选 mask，depth teacher 过滤几何不合理区域，再训练 Mean Teacher | 中-高 | 中 | foundation prior + pseudo label 已拥挤，容易被认为是组合式工程；depth 过滤需要强证据 | 可作为极少标注增强，不作为主创新 |

## 最推荐的 2026 CVPR 主线
| 排名 | 主线名称 | 组合来源 | 核心卖点 | 投稿判断 |
|---|---|---|---|---|
| 1 | Occlusion-Reflection Aware Depth-Privileged Mean Teacher | 4+5+6 | 把腹腔镜退化因素定义为 teacher reliability 与 occlusion geometry 问题，训练期用 depth 建模反光、烟雾、遮挡和接触边界，再蒸馏到 RGB-only student | 最强，临床问题明确，区别于普通 depth-guided mutual learning |
| 2 | Depth-Disentangled Prototype Mean Teacher | 6+3+4 | RGB 原型学习外观，depth 原型学习几何，训练期解耦并蒸馏，推理只保留 RGB 原型空间 | 强，技术表达清晰，适合做主要方法图和消融 |
| 3 | Temporal Occlusion Depth-Privileged Teacher | 9+5 | 用 depth 训练期识别遮挡和相邻帧几何变化，改善视频器械分割的伪标签传播 | 中-强，依赖连续帧数据和额外实验成本 |

## 不建议作为主创新的方向
`DG-CoTeach-MT`、`Depth-Guided Patch Curriculum MT`、`Ambiguity-Aware Surgical MT` 不建议单独作为主方案。前两者与 CVPR 2025 depth-guided mutual learning 路线接近，第三者与 CVPR 2025 AmbiSSL 路线接近。它们可以作为模块、baseline 或消融，但不能承担论文主要创新。

## 建议最终方案表述
推荐将论文主方案收敛为：`Occlusion-Reflection Aware Depth-Privileged Prototype Mean Teacher for RGB-only Laparoscopic Instrument Segmentation`。一句话定义：训练期使用 depth 作为特权几何模态，显式估计腹腔镜场景中的反光、烟雾、遮挡和器械-组织接触区域可靠性，通过几何原型蒸馏和局部一致性把 depth 先验转移到 RGB student，推理阶段只需要 RGB。

## 需要避免的审稿风险
| 风险 | 规避方式 |
|---|---|
| 被认为只是 RD-Net/DGML 的器械分割迁移 | 不以双学生互教或 depth patch 作为主创新，突出遮挡-反光可靠性建模和原型几何蒸馏 |
| 被认为只是 RGB-D 蒸馏旧问题 | 强调 depth 只训练期使用、半监督伪标签可靠性、腹腔镜临床退化因素，而不是普通 RGB-D 语义分割 |
| 被认为临床适配没有证据 | 按反光、烟雾、遮挡、尖端、小器械、低对比边界分组报告 Dice、NSD、HD95 和 failure case |
| 被认为 foundation model 拼接工程 | SAM/MedSAM 只放到附加实验或极少标注增强，不作为主线 |

## 参考方向
- CVPR 2025 `Boost the Inference with Co-training: A Depth-guided Mutual Learning Framework for Semi-supervised Medical Polyp Segmentation`
- CVPR 2025 `Annotation Ambiguity Aware Semi-Supervised Medical Image Segmentation`
- `SegMatch: a semi-supervised learning method for surgical instrument segmentation`
- `MedSAM: Segment Anything in Medical Images`
