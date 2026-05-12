# 深度学习代码审查报告

**项目**: ssl4mis/code_all — 腹腔镜手术器械半监督分割
**审查日期**: 2026-05-09
**审查范围**: 训练/测试全链路，12 个维度
**审查工具**: Claude DL Code Review Skill

---

## 一、审查结论总览

| 风险等级 | 数量 | 概述 |
|----------|------|------|
| CRITICAL | 0 | — |
| HIGH | 2 | 评估指标计算错误、EMA 模型模式切换 |
| MEDIUM | 8 | 数据划分、环境版本、模型代码、超参数配置等 |
| LOW | 9 | 可复现性、代码整洁性 |

**整体评价**：代码架构设计规范（策略注册 + 模型注册 + 分层参数），训练/验证/测试边界隔离正确，数据划分无泄露，归一化无泄露。主要问题集中在评估指标的边界处理和 EMA 教师模型的模式切换。

---

## 二、HIGH 级问题（2 个）

### 2.1 Precision/Recall 分母为零时返回 1.0，完全漏检被记为满分

- **文件路径 + 行号**: `core/test.py` L424-L425, L441-L442
- **风险等级**: HIGH

**问题定位**:
```python
'Precision': [item['TP'] / (item['TP'] + item['FP']) if (item['TP'] + item['FP']) > 0 else 1.0 ...]
'Recall':    [item['TP'] / (item['TP'] + item['FN']) if (item['TP'] + item['FN']) > 0 else 1.0 ...]
```

**违规学术原因**: 当模型对某类别完全没有预测（TP=0, FP=0）时返回 Precision=1.0。在半监督学习场景下（标注数据少，模型可能对某些类别欠学习），完全漏检某类会被记为满分，严重高估性能。

**整改方案**:
```python
# 修改前
'Precision': item['TP'] / (item['TP'] + item['FP']) if (item['TP'] + item['FP']) > 0 else 1.0
'Recall':    item['TP'] / (item['TP'] + item['FN']) if (item['TP'] + item['FN']) > 0 else 1.0

# 修改后
'Precision': item['TP'] / (item['TP'] + item['FP']) if (item['TP'] + item['FP']) > 0 else 0.0
'Recall':    item['TP'] / (item['TP'] + item['FN']) if (item['TP'] + item['FN']) > 0 else 0.0
```

**相关参考**: scikit-learn `precision_score` 的 `zero_division` 参数默认返回 0。

---

### 2.2 EMA 模型训练时处于 train 模式，导致 Dropout 激活

- **文件路径 + 行号**: `strategies/base_strategy.py` L141-L144
- **风险等级**: HIGH

**问题定位**:
```python
def _set_model_mode(self, training):
    self.model.train(mode=training)
    if self.ema_model is not None:
        self.ema_model.train(mode=training)  # BUG: EMA 应始终为 eval
```

**违规学术原因**: 标准 Mean Teacher 实践要求 EMA 教师模型始终处于 eval 模式，以确保教师输出稳定一致。当前实现中 UNet encoder 的 Dropout (0.05~0.5) 和 ResNet 的 Dropout2d 在 EMA 前向传播时仍然激活，破坏教师信号稳定性。影响所有使用 EMA 的策略（semi_mean_teacher、v1/v2/v3 等约 20 个策略）。

**整改方案**:
```python
# 方案 A：在 _set_model_mode 中强制 EMA 为 eval
def _set_model_mode(self, training):
    self.model.train(mode=training)
    if self.ema_model is not None:
        self.ema_model.eval()  # 始终保持 eval 模式

# 方案 B：在 _enable_ema_support 中初始化后不调 train()
```

**相关参考**: Mean Teacher (Tarvainen & Valpola, NeurIPS 2017) 原论文要求 teacher 使用 eval 模式的 BN/Dropout。

---

## 三、MEDIUM 级问题（8 个）

### 3.1 EndoVis 2018 val == test，模型选择和最终评估使用同一份数据

- **文件**: `data/endovis2018ISINet/val_slices.list` = `test_slices.list`（逐字节一致，596 条）
- **风险等级**: MEDIUM

**违规学术原因**: 模型选择（best checkpoint）和最终评估使用同一份数据，报告的 test 指标实际上就是 val 指标。审稿人可能质疑结果的泛化性。

**整改方案**: 划分独立的测试集，或在论文中明确说明 test=val 并解释原因。

---

### 3.2 semi_mt_depth_teacher_v1 的 validation_step 丢失 depth 输入

- **文件路径 + 行号**: `strategies/semi_mt_depth_teacher_v1.py` L136-L139
- **风险等级**: MEDIUM

**问题定位**:
```python
def validation_step(self, batch_data):
    with torch.no_grad():
        volume = batch_data["image"].to(self.device)
        return self.model(volume)  # 丢弃了 depth 拼接
```

**违规学术原因**: 覆盖了基类的 `validation_step`，但丢弃了 depth 拼接逻辑。如果模型以 4 通道（3 RGB + 1 depth）训练，验证时输入维度不匹配会导致运行时错误或静默产生错误结果。

**整改方案**: 参照基类实现，将 depth tensor 拼接到 volume 上。

---

### 3.3 PSNR 的 data_range 在 batch 级别计算

- **文件路径 + 行号**: `utils/metrics.py` L90-L92
- **风险等级**: MEDIUM

**问题定位**:
```python
data_min = torch.min(target)  # 整个 batch 的 min
data_max = torch.max(target)  # 整个 batch 的 max
data_range = (data_max - data_min).clamp(min=1e-6)
```

当 batch_size > 1 时，data_range 来自整个 batch 而非单个样本。训练验证阶段 batch_size=1 无影响，但测试阶段如果 batch_size > 1，不同样本的深度范围不同会导致 PSNR 计算不准确。

---

### 3.4 requirements.txt 全部使用 `>=`，无精确版本锁定

- **文件**: `requirements.txt`
- **风险等级**: MEDIUM

torch、timm 等核心依赖跨大版本 API 差异显著，不同时间安装可能得到不同版本，导致实验不可复现。无 environment.yml、Dockerfile、setup.py。

**整改方案**: 至少将 torch、torchvision、timm 固定到精确版本，或提供 requirements-lock.txt。

---

### 3.5 Python 版本未记录，代码使用 3.10+ 语法

- **文件**: `strategies/specs.py` L45-L47（使用 `str | None` 语法）
- **风险等级**: MEDIUM

无 .python-version 文件或 python_requires 声明。

---

### 3.6 RDNet/W2S 策略硬编码超参数，无法外部调参

- **文件路径 + 行号**: `strategies/semi_rdnet.py` L33-L37, `strategies/semi_w2s.py`
- **风险等级**: MEDIUM

`rdnet_thresh=0.85`、`contrastive_margin=0.5`、`contrastive_temp=0.07` 等超参数未通过 CLI 暴露，无法从外部调整，影响消融实验的灵活性。

---

### 3.7 优化器选择受限，只允许 Adam

- **文件路径 + 行号**: `core/args.py` L110, `core/train.py` L204
- **风险等级**: MEDIUM

`choices=["adam"]` 且 `betas/weight_decay` 硬编码。如果原论文基线使用 SGD，则对比实验不公平。

---

### 3.8 depth pretrain 策略 best checkpoint 选择用 PSNR 而非 Dice

- **文件路径 + 行号**: `core/train.py` L353-L359
- **风险等级**: MEDIUM

`_is_depth_pretrain_strategy()` 硬编码判断 `way == "fully_depth_pretrain_v1"`，该策略用 PSNR/SSIM 选 best，其他策略用 Dice。消融对比时需注意这一差异。

---

## 四、LOW 级问题（9 个）

| # | 问题 | 文件 | 说明 |
|---|------|------|------|
| 1 | pred=0,gt=0 时训练/测试 IoU 行为不一致 | `metrics.py:6` vs `test.py:265` | 训练返回 ~0.5，测试返回 1.0，建议统一 |
| 2 | val_loader 未设置 worker_init_fn | `core/train.py:172` | 推理无随机增强，影响极小 |
| 3 | 无跨运行 mean±std 统计 | — | 缺少多 seed 自动运行脚本，审稿人常见要求 |
| 4 | GPU 型号/CUDA 版本未记录 | README.md | 不同 GPU 架构浮点精度行为不同 |
| 5 | 监督损失权重硬编码 0.5*(CE+Dice) | 各策略文件 | 原论文可能使用不同权重，需确认 |
| 6 | EMA 初始化时调用 train() | `base_strategy.py:106` | 配合 HIGH-2 修复 |
| 7 | _SSIMLoss window 缓存不随设备迁移 | `losses.py:285` | 建议改用 register_buffer |
| 8 | Decoder_PROTO 含未使用的对比头参数 | `unet.py:118-119` | 浪费内存和 checkpoint 空间 |
| 9 | tools/ 中 setup_seed 不完整 | `tools/endovis2017_encoder_domain_gap.py:119` | 缺少 cudnn.deterministic/benchmark |

---

## 五、各维度审查详情

### 5.1 数据集划分与泄露 ✅ 通过（附注意事项）

- **序列级切分正确**：EndoVis 2017/2018 的 train/val 序列完全不相交
- **时序划分正确**：同一序列的帧不被拆分到不同集合
- **TwoStreamBatchSampler 无泄露**：labeled/unlabeled 索引不重叠
- **fold 交叉验证覆盖正确**：8 个序列每个恰好在 1 个 fold 做 val
- ⚠️ EndoVis 2018 val == test（MEDIUM-3.1）

### 5.2 预处理归一化泄露 ✅ 通过

- 归一化使用固定常数（ImageNet mean/std）或 per-sample 统计（minmax/255），无全局统计量泄露
- depth1 使用 dtype 上限（uint8→/255, uint16→/65535）而非数据集统计量
- train/val/test 共用同一归一化逻辑
- BatchNorm eval 模式切换正确

### 5.3 训练/验证/测试边界隔离 ✅ 通过

- 验证集不参与梯度更新（双层 `torch.no_grad()` 保护）
- 测试集在训练过程中完全不被访问
- `model.train()/eval()` 通过 `try/finally` 正确切换
- 半监督策略中 labeled/unlabeled 数据严格隔离

### 5.4 评估指标计算 ❌ 存在 HIGH 级问题

- ⚠️ **Precision/Recall 分母为零返回 1.0**（HIGH-2.1）
- Dice/IoU 除零保护正确（smooth=1e-6）
- 背景类正确排除（range(1, num_classes)）
- 宏平均聚合方式合理
- PSNR data_range 在 batch 级别计算（MEDIUM-3.3）

### 5.5 随机种子与可复现性 ✅ 通过

- `setup_seed()` 覆盖完整：torch/np/random/cudnn/PythonHashSeed
- train_loader 的 `worker_init_fn` 正确传播种子
- labeled/unlabeled 划分确定性，结果保存到 .list 文件
- ⚠️ 缺少跨运行 mean±std 自动聚合（LOW-3）

### 5.6 消融实验变量控制 ⚠️ 部分通过

- 策略注册中心设计合理，变量控制清晰
- ⚠️ RDNet/W2S 硬编码超参数（MEDIUM-3.6）
- ⚠️ 优化器只允许 Adam（MEDIUM-3.7）

### 5.7 超参数与训练配置 ✅ 通过

- Adam + PolyLR 是医学图像分割标准配置
- EMA 实现公式正确（warm-up + decay）
- 梯度裁剪正确实现

### 5.8 对比实验与基线 ⚠️ 部分通过

- 所有策略共享统一评估流程和数据增强管线
- 半监督基线实现规范（MT/UAMT/URPC/W2S）
- ⚠️ depth pretrain 策略 best checkpoint 用 PSNR 而非 Dice（MEDIUM-3.8）
- ⚠️ 监督损失权重硬编码，未与原论文对齐（LOW-5）

### 5.9 环境与版本 ⚠️ 存在 MEDIUM 级问题

- ⚠️ requirements.txt 全部 `>=`，无精确版本锁定（MEDIUM-3.4）
- ⚠️ 无 environment.yml/Dockerfile（MEDIUM-3.4）
- ⚠️ Python 版本未记录（MEDIUM-3.5）

### 5.10 代码质量 ⚠️ 存在问题

- ⚠️ EMA 模型 train/eval 模式切换错误（HIGH-2.2）
- ⚠️ validation_step 丢失 depth 输入（MEDIUM-3.2）
- 张量维度操作正确，梯度裁剪正确
- Decoder_PROTO 含未使用参数（LOW-8）

### 5.11 数据加载与增强泄露 ✅ 通过

- 训练用 `is_val=False`（随机旋转/翻转），验证/测试用 `is_val=True`（确定性）
- `_add_noise` 仅在训练的 `compute_loss` 中调用
- 无 Mixup/CutMix
- depth 与 RGB 使用同步随机变换

### 5.12 早停与模型选择泄露 ✅ 通过

- 最佳模型基于验证集选择
- 测试集仅评估一次
- 有验证时 final 模型被置空(0字节)，强制使用 best
- `--no_val` 模式有日志提示

---

## 六、整改优先级建议

| 优先级 | 操作 | 预计工作量 |
|--------|------|-----------|
| P0 | 修复 Precision/Recall 除零返回 1.0（HIGH-2.1） | 5 分钟 |
| P0 | 修复 EMA 模型 train/eval 切换（HIGH-2.2） | 10 分钟 |
| P1 | 修复 semi_mt_depth_teacher_v1 validation_step（MEDIUM-3.2） | 15 分钟 |
| P1 | 固定 requirements.txt 版本 + 添加 environment.yml（MEDIUM-3.4） | 30 分钟 |
| P2 | 确认 EndoVis 2018 val==test 并在论文中说明（MEDIUM-3.1） | 论文层面 |
| P2 | 将硬编码超参数暴露为 CLI 参数（MEDIUM-3.6） | 1 小时 |
| P3 | 其他 LOW 级改进 | 按需 |

---

*审查工具: Claude DL Code Review Skill*
