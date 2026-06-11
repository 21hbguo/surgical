# GeoRisk-SPC 方法详细说明

## 1. 问题定义

半监督手术器械分割旨在利用少量标注数据和大量未标注数据训练分割模型。现有方法主要面临以下挑战：
- 伪标签在边界和模糊区域不可靠
- 所有区域被同等对待，忽略区域差异
- 缺乏几何先验指导

## 2. 方法概述

GeoRisk-SPC 包含三个核心组件：
1. 几何感知风险定位 (Geometry-aware Risk Localization)
2. 区域感知监督 (Region-aware Supervision)
3. 结构扰动一致性 (Structural Perturbation Consistency)

## 3. 几何感知风险定位

### 3.1 局部相对深度归一化

给定深度图 $d_u$，我们进行局部归一化以消除全局深度偏差：

$$d_{rel} = \frac{d_u - \mu_{local}(d_u)}{\sigma_{local}(d_u) + \epsilon}$$

其中 $\mu_{local}$ 和 $\sigma_{local}$ 是滑动窗口内的局部均值和标准差。

### 3.2 深度不连续性

使用 Sobel 算子计算深度梯度：

$$G_d = \|\nabla d_{rel}\| = \sqrt{(\frac{\partial d_{rel}}{\partial x})^2 + (\frac{\partial d_{rel}}{\partial y})^2}$$

### 3.3 教师不确定性

计算教师模型预测的熵作为不确定性度量：

$$U_t = -\sum_{c=1}^{C} p_c \log p_c$$

其中 $p_c$ 是类别 $c$ 的预测概率。

### 3.4 几何-语义冲突

计算深度边界与语义边界的不一致性：

$$C_{conf} = |\text{norm}(G_d) - \text{norm}(B_p)$$

其中 $B_p$ 是预测边界图。

### 3.5 风险图计算

综合不确定性、深度不连续性和冲突信息：

$$R = \text{norm}(U_t) \cdot \text{norm}(G_d) + \lambda_c C_{conf}$$

区域划分：
- 高风险区域：$M_r = \mathbb{1}[R > \tau_r]$
- 低风险区域：$M_l = \mathbb{1}[\text{conf} > \tau_c] \cdot (1 - M_r)$

## 4. 区域感知监督

### 4.1 低风险伪标签损失

在低风险区域使用硬伪标签监督：

$$\mathcal{L}_{pl} = \frac{1}{|M_l|} \sum_{(i,j) \in M_l} \text{CE}(p_{clean}^{(i,j)}, \hat{y}^{(i,j)})$$

### 4.2 高风险一致性损失

在高风险区域使用软一致性约束：

$$\mathcal{L}_{cons} = \frac{1}{|M_r|} \sum_{(i,j) \in M_r} \text{KL}(p_{pert}^{(i,j)} \| p_{clean}^{(i,j)})$$

### 4.3 边界一致性损失

在高风险区域强制边界一致性：

$$\mathcal{L}_{bd} = \frac{1}{|M_r|} \sum_{(i,j) \in M_r} |\|\nabla p_{clean}\| - \|\nabla p_{pert}\||$$

### 4.4 总损失

$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_{pl}\mathcal{L}_{pl} + \lambda_{cons}\mathcal{L}_{cons} + \lambda_{bd}\mathcal{L}_{bd}$$

## 5. 结构扰动一致性

在瓶颈特征层应用风险引导的扰动：

$$f_{pert} = f_{clean} \cdot (1 - M_r \cdot D_{channel}) + M_r \cdot \mathcal{N}(0, \sigma^2)$$

其中 $D_{channel}$ 是通道级 dropout 掩码。

## 6. 模型架构

### 6.1 GeoRisk-SPC (Plain)
- 编码器：标准 UNet 编码器 (输入: 4ch RGB+Depth)
- 解码器：双解码器 (clean + perturbed)
- 扰动模块：RiskGuidedPerturbation

### 6.2 GeoRisk-SPC+DGv4
- 编码器：DepthGuiderV4 (输入: 3ch RGB，深度通过交叉注意力注入)
- 解码器：双解码器 (clean + perturbed)
- 扰动模块：RiskGuidedPerturbation

## 7. 实现细节

- 优化器：Adam (lr=1e-4, weight decay=1e-4)
- 批次大小：24 (12 labeled + 12 unlabeled)
- 训练迭代：30000
- 早停：patience=9000 iterations
- 深度图：预计算的单目深度估计 (1ch)
- 数据增强：弱增强 (教师) + 强增强 (学生)
