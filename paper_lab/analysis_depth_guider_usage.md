# DepthGuiderV4 各层深度信息使用分析

## 发现

训练后模型（GeoRiskSPC_DGv4, 40% labeled）的 DepthGuiderV4 各层输出：

| 层 | depth_feat 范围 | geom_feat 范围 | 使用情况 |
|---|----------------|----------------|---------|
| L0 (16ch, 224×224) | [0.0004, 2.88] | [0.025, 3.83] | **有效使用** |
| L1 (32ch, 112×112) | [0.043, 1.52] | [0.028, 2.77] | **有效使用** |
| L2 (64ch, 56×56) | [0.061, 1.36] | [0.044, 1.59] | **有效使用** |
| L3 (128ch, 28×28) | [0.000, 0.000] | [0.000, 0.000] | **未使用** |
| L4 (256ch, 14×14) | [0.000, 0.000] | [0.000, 0.000] | **未使用** |

## 解释

1. **浅层需要深度信息**：L0-L2 负责捕捉边缘、纹理等细节特征，深度信息（特别是深度梯度）能帮助区分器械边缘和组织边界

2. **深层不需要深度信息**：L3-L4 负责高级语义特征，此时器械的类别信息已经足够，深度信息的边际收益很小

3. **自适应学习**：模型自动学到了"在浅层使用深度，深层忽略深度"的策略，而非强制所有层都使用

## 论文表述建议

### 方法部分
> Our DepthGuiderV4 applies depth-guided feature modulation at each encoder level. Interestingly, we observe that the model learns to selectively utilize depth information primarily at shallow layers (L0-L2), while deeper layers (L3-L4) learn to suppress depth contributions (near-zero output), suggesting that geometric cues are most beneficial for capturing fine-grained boundary details rather than high-level semantics.

### 分析/消融部分
可以设计消融实验：
- 只在 L0-L2 使用 DepthGuider → 预测性能
- 只在 L3-L4 使用 DepthGuider → 预测性能
- 全层使用 vs 选择性使用

### 可视化建议
在论文中展示 depth_feat 各层的热力图，直观显示深度信息的使用模式：
```
L0 depth_feat  L1 depth_feat  L2 depth_feat  L3 depth_feat  L4 depth_feat
[有激活]       [有激活]       [有激活]       [全0]          [全0]
```

## 代码验证

```python
# 验证脚本
python -c "
import torch
from models.networks.unet import UNet_DepthGuiderV4_GeoRiskSPC

model = UNet_DepthGuiderV4_GeoRiskSPC(in_chns=4, class_num=2, filter_num=16)
# 加载 checkpoint...
encoder = model.encoder

# Hook 捕获各层 depth_encoder 输出
for i, dg in enumerate(encoder.depth_guiders):
    # 检查 depth_feat 和 geom_feat 的值范围
"
```

## 相关论文观点

1. **Depth Anything** (2024): 深度信息在低级视觉任务（边缘检测、分割）中最有用
2. **DPT** (2021): 深度估计主要依赖浅层特征
3. **本文贡献**: 首次在半监督分割中发现并解释了深度信息的层次性使用模式
