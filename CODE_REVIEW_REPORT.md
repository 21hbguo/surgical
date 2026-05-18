# SSL4MIS 深度学习训练代码审计报告

> 审查范围: `core/`, `strategies/`, `data/`, `utils/`, `models/`
> 审查日期: 2026-05-18
> 审查维度: 实验合规、训练逻辑、数据流程、损失函数、优化器配置、正则化、梯度问题、数据集划分、标签泄露、参数合理性、显存效率、日志断点、半监督训练规范

---

## 一、关键问题 (CRITICAL)

### 1.1 NTXentLoss 标签约标错误 —— 对比学习监督信号错误

**文件**: `utils/losses.py:18`

```python
labels = torch.arange(z_i.shape[0], device=z_i.device)
labels = torch.cat([labels + z_i.shape[0] - 1, labels], dim=0)
```

**问题**: 去除对角线后，相似度矩阵的正样本对索引映射存在 off-by-one 错误。当 `N = z_i.shape[0]` 时，第二半部分标签应为 `labels + N`（去掉 `-1`），否则正样本对指向错误位置。

**正确推导**:
- 原始相似度矩阵 `sim` 形状 `[2N, 2N]`
- 去除对角线后 `sim[~mask]` 形状 `[2N, 2N-1]`，reshape 为 `[2N, 2N-1]`
- 第 `i` 行的原始列 `j` 在新矩阵中的索引: `j < i` → `j`, `j > i` → `j - 1`
- 对于 `i ∈ [0, N)`（z_i 部分），正样本在 `z_j` 中的原始索引为 `i + N`
- 新索引 = `i + N - 1`（因 `i + N > i`，需减 1）
- 对于 `i ∈ [N, 2N)`（z_j 部分），正样本在 `z_i` 中的原始索引为 `i - N`
- 新索引 = `i - N`（因 `i - N < i`，不需调整）

**验证** (N=2, 4 个样本):
```
第二半部分 (i=2,3):
  i=2 → 正样本原始列=0 → 新索引=0 ✓, 标签=0 ✓
  i=3 → 正样本原始列=1 → 新索引=1 ✓, 标签=1 ✓

第一半部分 (i=0,1):
  i=0 → 正样本原始列=2 → 新索引=2-1=1 ✓, 标签=0+2-1=1 ✓
  i=1 → 正样本原始列=3 → 新索引=3-1=2 ✓, 标签=1+2-1=2 ✓
```

**结论**: 经过详细推导，该标签构造实际上是**正确的**。`labels + z_i.shape[0] - 1` 与 `i + N - 1` 一致。**此项不是 bug**。

### 1.2 model_final.pth 在有验证时保存空文件

**文件**: `core/train.py:441-444`

```python
def _save_model(self, suffix):
    path = os.path.join(self.snapshot_path, f"model_{suffix}.pth")
    if suffix == "final" and self.has_val:
        with open(path, "wb"):
            pass  # 创建 0 字节文件
        return
```

**问题**: 当存在验证集时，`model_final.pth` 被创建为 0 字节空文件。这意味着:
1. **丢失最终模型**: 如果最佳模型发生在 iter 20000，而训练持续到 iter 30000，最终迭代的模型权重完全丢失。
2. **测试时无法加载**: `core/test.py:554` 尝试 `load_checkpoint(model, checkpoint_path)` 会因空文件而失败。
3. **部分检查点恢复失效**: `core/runtime.py` 的 `existing_train_checkpoints()` 会认为训练已完成（因为文件存在），从而跳过该 fold。

**严重程度**: 高。虽然设计意图是"有 best 就够了"，但 `model_final.pth` 的存在会误导跳过逻辑，且 `--checkpoint-type final` 会直接失败。

### 1.3 sigmoid_rampup 参数名不一致

**文件对比**:

| 文件 | 签名 |
|------|------|
| `utils/common.py:30` | `sigmoid_rampup(current, rampup_length)` |
| `utils/losses.py:119` | `sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0)` |

**调用点**:

- `base_strategy.py:56`: `sigmoid_rampup(iter_num // div, rampup)` — 2 个位置参数，匹配 `utils/common.py` 版本
- `semi_uncertainty_mt.py:66`: `sigmoid_rampup(iter_num, self.max_iter)` — 2 个位置参数，匹配 `utils/common.py` 版本
- `utils/losses.py:192-196` (FeCLoss): `sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5)` — 5 个参数，匹配 `utils/losses.py` 版本

**结论**: 两个同名函数存在于不同模块，功能不同。调用点各自使用正确的版本，不会产生运行时错误。但这是维护隐患——未来修改时容易混淆。

---

## 二、高危问题 (HIGH)

### 2.1 训练中断后不支持断点续训

**文件**: `core/train.py:495-501`

```python
if existing_checkpoints:
    logging.info("Partial checkpoints found: %s, continuing training on %s", ...)
    # 实际上从 iter_num=0 重新开始，并不恢复状态
```

**问题**: 当检测到部分检查点时，代码记录日志后从头开始训练，并不加载已有权重。这意味着:
- 用户以为在"继续训练"，实际从零开始
- 之前消耗的 GPU 时间全部浪费
- 如果 `model_best.pth` 存在但 `model_final.pth` 不存在，会覆盖已有的 best 模型

### 2.2 EMA 模型创建时处于 train 模式

**文件**: `base_strategy.py:109`

```python
def _enable_ema_support(self):
    ...
    self.ema_model.train()  # 显式设为 train 模式
```

**问题**: EMA 教师模型创建后被设为 `train()` 模式。虽然后续 `_set_model_mode()` 会将其强制设为 `eval()`，但在以下场景可能出问题:
1. 如果某个策略在 `_set_model_mode` 调用前就使用 EMA 模型推理
2. 如果模型中存在 `Dropout` 层，train 模式下的 dropout 会导致教师输出不稳定

**影响**: 在当前代码中，所有策略的 `compute_loss` 都在 `training_step` 中调用，而 `training_step` 之前会调用 `_set_model_mode(True)` → EMA 设为 eval。所以**当前不会触发**，但属于防御性编程缺失。

### 2.3 UncertaintyMT: sigmoid_rampup 参数语义错误

**文件**: `semi_uncertainty_mt.py:66`

```python
threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, self.max_iter)) * np.log(2)
```

这里调用的是 `utils/common.py` 的 `sigmoid_rampup(current, rampup_length)`，该函数返回 `[0, 1]` 范围的值。所以 threshold 范围为 `[0.75 * ln2, 1.0 * ln2] ≈ [0.52, 0.69]`。

**但** `uncertainty` 是信息熵，对于二分类最大值为 `ln(2) ≈ 0.693`。初始 threshold = `0.75 * ln2 ≈ 0.52` 意味着**一开始就会过滤掉约 25% 的中等不确定度像素**，这可能是过于激进的。

### 2.4 部分策略的 get_state_dict 未保存 EMA 模型

**文件**: `base_strategy.py:166-167`

```python
def get_state_dict(self):
    return {"model": self.model.state_dict()}
```

**问题**: 基类只保存 student 模型。对于需要教师模型的策略（如 Proto、DepthGuider-Proto-Teacher），EMA 模型的权重不被保存。这意味着:
- 测试时无法加载教师模型进行集成预测
- 如果训练中断，教师模型的累积知识完全丢失
- 但这是有意设计——EMA 可以从 student 重建

### 2.5 GradScaler 多优化器 step 顺序问题

**文件**: `semi_mt_depth_guider_proto_teacher_v3.py:395-400`

```python
self.scaler.step(self.optimizer)
self.scaler.step(self.depth_teacher_optimizer)
self.scaler.step(self.appearance_teacher_optimizer)
self.scaler.step(self.proto_optimizer)
self.scaler.step(self.disentangle_optimizer)
self.scaler.update()  # 只调用一次 update
```

**问题**: PyTorch 官方文档建议在多个 `scaler.step()` 后只调用一次 `scaler.update()`，这是正确的。但如果某个优化器的梯度包含 `inf/NaN`，`scaler.step()` 会跳过该优化器的更新，但后续优化器仍会正常更新。这可能导致模型组件之间的训练不同步。

---

## 三、中危问题 (MEDIUM)

### 3.1 DiceLoss 的 target 平方计算

**文件**: `utils/losses.py:82`

```python
def _dice_loss(self, score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)  # 而非 torch.sum(target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - loss
```

**问题**: `y_sum = torch.sum(target * target)` 对于 one-hot 编码（值为 0 或 1），结果等价于 `torch.sum(target)`。所以数学上等价，但使用 `target * target` 会增加不必要的计算和显存开销（虽然微小）。这是从原始 Dice loss 实现继承来的写法。

### 3.2 数据增强强度有限

**文件**: `data/transforms.py:99-148`

**问题**:
1. **缺少颜色增强**: 没有亮度、对比度、饱和度抖动。对于腹腔镜手术场景，光照变化是重要的域偏移来源。
2. **缺少弹性变形**: 仅使用旋转/翻转/小角度旋转，缺少弹性变形、网格变形等非刚性增强。
3. **固定 resize 到 224x224**: 对于高分辨率医学图像（如 640x480 的 EndoVis），下采样到 224x224 会丢失大量细节。
4. **验证集也 resize**: `val_loader` 使用与训练相同的 `resize_size`，但测试时保留原始分辨率。验证指标和测试指标可能不一致。

### 3.3 优化器配置固定

**文件**: `core/train.py:213`

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
```

**问题**:
1. **不支持 AdamW**: Adam 和 AdamW 的 weight decay 实现不同。对于 Transformer 模型（DINOv3, DFormerv2），AdamW 通常更优。
2. **不支持差异化学习率**: 预训练 backbone（如 DINOv3）通常需要比 decoder 更低的学习率，但当前所有参数共享同一 lr。
3. **固定 betas**: 不允许用户调整动量参数。

### 3.4 学习率调度器无最低学习率保护

**文件**: `utils/lr_scheduler.py:18`

```python
lr = max(self.base_lr * ((1 - progress) ** self.power), self.min_lr_ratio * self.base_lr)
```

默认 `min_lr_ratio=0`，意味着训练末期 lr 可以降到 0。虽然 `args.py` 默认值为 0，但实际调用时 `args.lr_min_ratio` 默认为 0（`args.py:155`）。lr=0 会导致最后大量迭代完全不更新参数。

### 3.5 测试指标空样本处理

**文件**: `core/test.py:257-266`

```python
if pred.sum() == 0 and gt.sum() == 0:
    return {
        'Dice': 1.0,  # 两个都为空时 Dice=1
        'IoU': 1.0,   # 两个都为空时 IoU=1
        ...
    }
```

**问题**: 当某个类在 pred 和 gt 中都不存在时，返回 Dice=1.0。这会人为拉高平均指标。在医学图像中，如果某些类在特定序列中很少出现，这些"完美"的空预测会掩盖真实性能。建议用 NaN 标记并在汇总时排除。

### 3.6 混合精度训练中 GradScaler 可能导致训练不稳定

**文件**: `base_strategy.py:130-138`

**问题**: 当 AMP 开启且 loss 包含多个损失项（如 CE + Dice + consistency + proto + geometry_align + mi_loss）时，某些损失项可能产生较大的梯度值，导致 GradScaler 频繁跳过更新。建议在复杂策略中默认开启梯度裁剪。

### 3.7 DataLoader 配置

**文件**: `core/train.py:167-173`

```python
train_loader = DataLoader(
    train_dataset,
    batch_sampler=batch_sampler,
    num_workers=args.num_workers,  # 默认 2
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)
```

**问题**:
1. **num_workers=2 较低**: 对于多核服务器，2 个 worker 可能成为数据加载瓶颈。建议默认 4-8。
2. **缺少 persistent_workers**: 每个 epoch 结束后 worker 会重启，增加开销。
3. **缺少 prefetch_factor**: 默认 prefetch_factor=2，可根据显存情况调大。

---

## 四、低危问题 (LOW)

### 4.1 日志和调试

1. **proto_v1 策略中残留 `print` 语句**: `strategies/semi_proto_v1.py:65-66` 使用 `print()` 而非 `logging`，不统一。
2. **训练 CSV 日志无锁**: `utils/save_vars_to_csv.py` 写 CSV 时无文件锁，多 fold 并行训练时可能冲突。
3. **无 TensorBoard/WandB 集成**: 仅使用 CSV + matplotlib 绘图，缺少实时监控能力。

### 4.2 代码风格和维护

1. **策略间代码重复**: `semi_mt_depth_guider_v1/v2/v3` 之间大量重复代码，应提取公共基类。
2. **_create_branch_model 重复**: `base_strategy._create_ema_model` 和 `v3._create_branch_model` 逻辑几乎相同。
3. **hardcoded 魔法数字**: `0.5 * (loss_dice + loss_ce)` 中的 0.5 权重、`0.1` 噪声标准差等应可配置。

### 4.3 测试流程

1. **测试时逐样本推理**: `core/test.py:325` 对 batch 中每个样本单独构建 `sample_batch`，未利用 batch 并行。
2. **Grad-CAM 重复前向**: 每个类别单独计算一次 Grad-CAM，可合并为单次前向+多次反向。
3. **序列号解析脆弱**: `core/test.py:386-393` 用正则从 case name 中提取 seq id，依赖命名约定。

---

## 五、数据流程审计

### 5.1 数据泄露检查 ✅ 无泄露

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Train/Val 分离 | ✅ | 使用不同 .list 文件 |
| Test 与 Train 分离 | ✅ | test_slices.list 独立 |
| 半监督标签泄露 | ✅ | labeled/unlabeled 均来自 train split |
| 增强一致性 | ✅ | image/label/depth 同步变换 |
| 标签插值 | ✅ | 使用 nearest-neighbor 保留离散值 |
| `use_val=True` 路径 | ✅ | 代码中存在但未激活 |

### 5.2 数据集划分

- **划分方式**: 预生成的 .list 文件 + JSON fold 映射
- **Fold 过滤**: 测试时通过 `fold_map` 过滤序列 ID
- **Labeled/Unlabeled 分割**: 基于 index 顺序（前 N 个为 labeled）
- **采样策略**: 支持 `interval` 均匀采样和 `none` 顺序取前 N

### 5.3 数据预加载

**文件**: `data/dataset.py:223-252`

使用 `ProcessPoolExecutor(max_workers=4)` 并行加载所有图片到内存。优点是训练时无 IO 瓶颈，缺点是:
1. 大数据集可能导致内存不足
2. 预加载失败时错误信息不够清晰
3. 进程池创建/销毁有开销

---

## 六、损失函数审计

### 6.1 各策略损失组合

| 策略 | 监督损失 | 一致性损失 | 对比损失 | 其他 |
|------|----------|-----------|----------|------|
| fully | 0.5*(CE+Dice) | - | - | - |
| mt | 0.5*(CE+Dice) | MSE * sigmoid_rampup | - | - |
| urpc | 0.5*(CE+Dice) | KL * uncertainty | - | 多尺度深监督 |
| proto_v1 | 0.5*(CE+Dice) | MSE * rampup | NTXent | - |
| dycon | 0.5*(CE+Dice) | UnCLoss | FeCLoss | - |
| w2s | 0.5*(CE+Dice) | decoder 一致性 | - | 特征扰动 |
| fully_contrast_v1 | 0.5*(CE+Dice) | - | boundary-guided | - |
| depth_guider_proto_teacher_v3 | 0.5*(CE+Dice + depth_CE + depth_Dice) | MSE*3 (EMA+depth+appearance) | NTXent | geometry_align + geometry_invariance + mi_loss |

### 6.2 损失权重调优建议

- **v3 策略**: 总损失包含 **10+ 个损失项**，权重调优空间巨大。建议:
  - 使用 uncertainty weighting 或 GradNorm 自动平衡
  - 记录每项损失的实际值到日志，便于诊断主导损失
  - 考虑分阶段训练（先监督，再一致性，最后对比）

### 6.3 潜在数值问题

1. **NTXentLoss 温度 0.07**: 较低的温度会放大 logits，可能导致数值溢出。当特征维度高时需注意。
2. **FeCLoss 的 1e-18 smoothing**: 在 float16 下 `1e-18` 可能下溢为 0。AMP 训练时有风险。
3. **UnCLoss 的 exp(beta * H)**: 当 beta 较大且 entropy 较高时，`exp()` 可能溢出。

---

## 七、优化器与训练配置审计

### 7.1 Adam 配置

```
Optimizer: Adam
LR: 1e-4 (默认)
Betas: (0.9, 0.99)
Weight Decay: 0.0001
```

**评价**: 对于 UNet 类模型，Adam + lr=1e-4 是合理默认值。但对于:
- **DINOv3 backbone**: 建议 lr=1e-5 ~ 5e-5（预训练特征需保守更新）
- **DFormerv2**: 建议使用 AdamW + cosine schedule
- **Prototype 学习**: 当前与主模型使用相同 lr，可能导致原型更新过快

### 7.2 学习率调度

```
Scheduler: Polynomial Decay
Power: 0.9
Warmup: 0 iters (默认无 warmup)
Min LR Ratio: 0 (可降到 0)
```

**问题**:
1. **无 warmup**: 对于预训练模型微调，warmup 可防止初始阶段梯度爆炸
2. **Min LR = 0**: 训练末期完全停止学习，浪费计算资源
3. **Poly power 0.9**: 衰减较慢，30000 次迭代后 lr ≈ 0.37 * base_lr

### 7.3 Early Stopping

```
Patience: 20% * max_iterations = 6000 iters (默认)
```

**评价**: 合理。但需要注意:
- Early stopping 基于验证 Dice，如果验证集很小，Dice 波动大可能导致过早停止
- 建议增加 `min_iterations` 参数，确保最少训练一定迭代数

---

## 八、半监督训练规范审计

### 8.1 Mean Teacher 框架

| 检查项 | 状态 | 说明 |
|--------|------|------|
| EMA 更新 | ✅ | warmup + decay，alpha=min(1-1/(step+1), ema_decay) |
| EMA eval 模式 | ⚠️ | 创建时 train()，但使用前会被设为 eval() |
| 噪声注入 | ✅ | student/teacher 分别注入不同噪声 |
| 一致性权重 rampup | ✅ | sigmoid rampup 防止早期过拟合伪标签 |
| 标签泄露 | ✅ | labeled/unlabeled 严格分离 |

### 8.2 伪标签质量控制

| 策略 | 过滤方式 | 评价 |
|------|----------|------|
| Mean Teacher | MSE 软标签（无过滤） | ✅ 标准做法 |
| UncertaintyMT | 熵阈值过滤 | ✅ 合理 |
| Proto | 熵分位数过滤 | ✅ 合理 |
| v3 | EMA + depth teacher 双重一致性 | ✅ 保守策略 |

### 8.3 多教师框架 (v3)

**架构**: Student + EMA Teacher + Depth Teacher + Appearance Teacher

**潜在问题**:
1. **4 个模型同时在 GPU**: 显存占用约为单模型的 4 倍
2. **5 个优化器**: 参数更新路径复杂，难以调试
3. **Depth Teacher 和 Appearance Teacher 都从 Student 初始化**: 初始阶段三者输出几乎相同，一致性损失可能退化为自蒸馏

---

## 九、显存效率审计

### 9.1 模型显存

| 组件 | 估计显存 (224x224, UNet) |
|------|-------------------------|
| Student 模型 | ~100 MB |
| EMA Teacher | ~100 MB |
| Depth Teacher (v3) | ~100 MB |
| Appearance Teacher (v3) | ~100 MB |
| Prototype bank | ~1 MB |
| Optimizer states (×5) | ~600 MB |
| **v3 策略总计** | **~1 GB (仅模型)** |

### 9.2 数据显存

- 输入: `[4, 4, 224, 224]` (4 通道 RGB+depth) ≈ 3 MB/batch
- 标签: `[4, 224, 224]` ≈ 0.8 MB/batch
- 中间特征: 取决于模型深度，通常 100-500 MB

### 9.3 优化建议

1. **EMA 模型使用 float16**: EMA 教师不需要 float32 精度
2. **梯度检查点**: 对于深 encoder，可使用 `torch.utils.checkpoint` 节省中间激活显存
3. **Prototype bank 可用 register buffer**: 避免优化器状态开销

---

## 十、复现性审计

### 10.1 种子控制 ✅

```python
# utils/common.py:163-171
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
```

### 10.2 Worker 种子 ✅

```python
# core/train.py:112-118
def worker_init_fn(worker_id):
    seed = args.seed + worker_id
    # 设置所有 RNG 种子
```

### 10.3 Sampler 种子 ✅

```python
# data/samplers.py:31
rng = np.random.RandomState(self.seed + self.epoch)
```

### 10.4 复现性风险

1. **torch.compile**: 可能引入非确定性编译优化
2. **AMP**: float16 累加顺序不确定，不同 GPU 架构结果可能不同
3. **多进程数据加载**: `ProcessPoolExecutor` 的任务分配顺序不确定（但结果缓存后不影响训练）

---

## 十一、总结与优先级建议

### 必须修复 (P0)

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 1 | model_final.pth 保存空文件 | `core/train.py:441-444` | 测试时加载失败 |
| 2 | 断点续训实际不恢复 | `core/train.py:495-501` | 中断后浪费全部计算 |

### 建议修复 (P1)

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| 3 | EMA 模型创建时 train 模式 | `base_strategy.py:109` | 潜在 Dropout 影响 |
| 4 | sigmoid_rampup 命名不一致 | `utils/` 两个版本 | 维护隐患 |
| 5 | 空 Dice=1.0 虚高指标 | `core/test.py:257-266` | 测试指标不准 |
| 6 | Min LR=0 浪费末期计算 | `utils/lr_scheduler.py` | 训练效率低 |
| 7 | 无 warmup | `utils/lr_scheduler.py` | 预训练模型微调不稳 |

### 可选优化 (P2)

| # | 问题 | 影响 |
|---|------|------|
| 8 | 数据增强缺少颜色变换 | 域泛化能力弱 |
| 9 | 不支持差异化学习率 | 预训练 backbone 训练不优 |
| 10 | 无 TensorBoard 集成 | 实验监控不便 |
| 11 | 测试时逐样本推理 | 测试速度慢 |
| 12 | num_workers=2 偏低 | 数据加载瓶颈 |

---

## 附录: 各文件审查行数统计

| 文件 | 行数 | 问题数 |
|------|------|--------|
| `core/train.py` | 541 | 4 |
| `core/test.py` | 651 | 3 |
| `core/args.py` | 188 | 1 |
| `strategies/base_strategy.py` | 173 | 2 |
| `strategies/semi_mt_depth_guider_proto_teacher_v3.py` | 450 | 2 |
| `strategies/semi_proto_v1.py` | 221 | 1 |
| `strategies/semi_uncertainty_mt.py` | 85 | 1 |
| `strategies/semi_mean_teacher.py` | 60 | 0 |
| `utils/losses.py` | 415 | 2 |
| `utils/metrics.py` | 104 | 0 |
| `utils/lr_scheduler.py` | 54 | 2 |
| `utils/common.py` | 252 | 1 |
| `data/dataset.py` | 331 | 1 |
| `data/transforms.py` | 149 | 1 |
| `data/samplers.py` | 66 | 0 |
| **合计** | **3841** | **21** |
