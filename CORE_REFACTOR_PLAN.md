# core/ 重构方案

## 原则
- **不拆分文件**：训练/测试主线必须在一个文件里一眼可见
- **消除重复**：只合并真正重复的代码，不做"优化"、"简化"
- **不新增包装**：所有修改都是删除或合并，不新增 class/function

## 问题清单（按 AGENTS.md 标准）

### 问题 1：train_h5.py 与 train.py 95% 重复（最严重）

**文件**：`core/train_h5.py`（186 行）vs `core/train.py`（516 行）

重复块：
- `setup_logging`：train.py:445-456 vs train_h5.py:129-139
- `_repeat_to_min_length`：train.py:42-47 vs train_h5.py:26-31
- `worker_init_fn`：train.py:105-111 vs train_h5.py:81-87
- labeled/unlabeled 索引计算：train.py:113-143 vs train_h5.py:89-119
- DataLoader 构建：train.py:145-152 vs train_h5.py:121-126
- `main()` fold 循环：train.py:459-512 vs train_h5.py:142-186

唯一差异：`BaseDataSets` vs `H5DataSets`，用 `--data-format png|h5` 参数一行切换

**monkey-patch 反模式**：train_h5.py:163-169
```python
import core.train
original_create = core.train.create_dataloaders
core.train.create_dataloaders = create_dataloaders_h5  # 运行时替换
```
违反 AGENTS.md 规则 7"禁止钩子式跳转"

**方案**：
1. `core/args.py` 的 `add_train_args` 增加 `--data-format` 参数
2. `core/train.py` 的 `create_dataloaders` 中按 `args.data_format` 选择 `BaseDataSets` 或 `H5DataSets`
3. 删除 `core/train_h5.py`
4. 入口统一为 `python -m core.train --data-format h5 ...`

### 问题 2：Trainer 类让训练主线不可见（核心问题）

**文件**：`core/train.py:205-443`（239 行 `Trainer` 类）

AGENTS.md 规则 1 明确说"不写通用训练框架"，规则 2 说"训练主线一眼可见"。当前 `main()` 里只能看到 `trainer.train()` 一个调用，实际的 `model→optimizer→loss→train→validate→save` 流程被藏在 Trainer 类的 10 个方法里：

```
main() → trainer.train()  # 看不到主线
```

应该变成：
```
main() → seed → folds → dataloader → model → optimizer → strategy/loss → train_epoch → validate → save  # 一眼可见
```

**方案**：将 `Trainer.train()`、`Trainer._train_epoch()`、`Trainer._validate_and_save()`、`Trainer._save_model()` 的逻辑内联到 `main()` 的 fold 循环中。删除 `Trainer` 类、`TrainComponents` dataclass、`build_train_components`。

### 问题 3：test_gamma_beta.py 不属于 core/

**文件**：`core/test_gamma_beta.py`（533 行）

不是训练/测试主线，是专用的 depth guider 可视化分析工具，混在 core/ 中模糊主线边界。

**方案**：`mv core/test_gamma_beta.py tools/depth_guider_analysis.py`

### 问题 4：calculate_metric_percase 重复实现

**文件**：`core/test.py:212-233` vs `utils/metrics.py:40-80`

功能完全重复（Dice/IoU/TP/FP/FN/Acc/Valid），utils/metrics 版本更全面（还包含 Class 字段）。

**方案**：删除 `test.py:calculate_metric_percase`，调用改为 `utils.metrics.calculate_segmentation_case_metrics`

### 问题 5：metric pairs 常量重复

**文件**：`core/test.py:37-43` vs `core/testing/export.py:20-26`

byte-identical 的 `(('Dice', 'dice'), ('IoU', 'iou'), ...)` 常量。

**方案**：将常量移到 `utils/metrics.py`（已有 metrics 相关逻辑），两处 import 引用

### 问题 6：参数重复解析

**文件**：`core/test_gamma_beta.py:34-38` `create_inference_strategy`

用 `build_train_parser().parse_args(["--task", ...])` 重新解析 args，然后逐个 setattr 复制。违反规则 5"参数只解析1次"。

**方案**：随问题 3 一起移出 core/ 后，在 tools 文件中修改（工具文件不受 AGENTS.md 严格约束）

## 执行顺序

| 步骤 | 操作 | 风险 | 验证 |
|------|------|------|------|
| 1 | 移动 `test_gamma_beta.py` → `tools/` | 零 | 确认无外部 import |
| 2 | 统一 metric pairs 常量 | 低 | pytest |
| 3 | 替换 `calculate_metric_percase` | 低 | 对比指标值一致 |
| 4 | 合并 `train_h5.py` 到 `train.py` | 中 | H5 数据训练验证 |
| 5 | 内联 `Trainer` 类到 `main()` | 高 | 完整训练验证 |

每步后 `pytest tests/ -q`。
