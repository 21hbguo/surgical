# SSL4MIS 代码质量与开源就绪度审查报告

> 审查日期：2026-05-18
> 审查范围：全部 117 个 Python 文件 + 6 个 Shell 脚本 + 配置/文档文件
> 另见：[CODE_REVIEW_REPORT.md](CODE_REVIEW_REPORT.md)（训练逻辑与学术规范审查）

---

## 一、总评

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | ★★★★☆ | 策略注册表 + 模型工厂模式，扩展性优秀 |
| 代码规范 | ★★★★☆ | 整体整洁，无 TODO/FIXME，结构清晰 |
| 可移植性 | ★★☆☆☆ | 60+ 处硬编码绝对路径，9 个中文文件名 |
| 测试覆盖 | ★★★☆☆ | 22 个测试文件覆盖核心逻辑，缺 loss/metrics 测试 |
| 开源就绪度 | ★★☆☆☆ | 缺 LICENSE、英文 README、打包配置 |

---

## 二、必须修复（开源阻断项）

### 2.1 缺少 LICENSE 文件

无任何许可证文件。开源项目必须明确授权协议，否则代码默认受版权保护，他人无法合法使用。

**建议：** 根据项目需求选择 MIT / Apache-2.0 / BSD-3-Clause。

### 2.2 README 仅有中文

当前 `README.md` 全为中文（185 行），国际用户无法使用。

**建议：** 提供英文 README（可保留中文版为 `README_CN.md`），包含：
- 项目简介与核心贡献
- 安装步骤
- 快速开始命令
- 策略/模型列表
- 数据准备说明
- BibTeX 引用格式

### 2.3 硬编码绝对路径（60+ 处）

这是最严重的可移植性问题。`/home/guo/...` 路径遍布项目，其他机器上直接报错。

**核心训练/测试代码（2 处）：**

| 文件 | 行号 | 硬编码内容 |
|------|------|-----------|
| `tests/test_task_dataset_selection.py` | 80 | 策略文件绝对路径 |
| `tests/test_task_dataset_selection.py` | 199 | `result_predict` 目录路径 |

**tools/ 目录（14 个文件，50+ 处）：**

| 文件 | 硬编码数量 | 内容类型 |
|------|-----------|---------|
| `tools/v2_auto_iterate.py` | 3 | 项目目录、进度 JSON |
| `tools/注意力可视化.py` | 6 | 权重路径、仓库路径 |
| `tools/感受野可视化.py` | 6 | 权重路径、仓库路径 |
| `tools/感受野.py` | 2 | 权重路径 |
| `tools/域解释.py` | 12 | 数据切片 + 权重 + 输出 |
| `tools/endovis2017_seq_video_builder.py` | 9 | 图像/标签/深度/输出 |
| `tools/parts_problem_mask.py` | 2 | 原始数据路径 |
| `tools/endovis2017_task2_concat.py` | 1 | 结果目录 |
| `tools/深度图单通道转三通道.py` | 2 | 深度图目录 |
| `tools/task3_label_remap.py` | 4 | 原始数据路径 |
| `tools/endovis_interp_compare.py` | 3 | 数据路径 |
| `tools/endovis_panel_stitcher.py` | 2 | 数据路径 |
| `tools/endovis2017_encoder_domain_gap.py` | 3 | 数据/checkpoint |
| `tools/diffslic_benchmark.py` | 3 | 文件路径 |

**建议：**
- 核心代码（`core/`, `data/`, `models/`, `strategies/`）：已是相对路径，无需修改
- 测试代码：用 `tmp_path` fixture 或环境变量替代
- tools/：改为 argparse 参数或从环境变量读取

### 2.4 中文文件名（9 个）

| 当前文件名 | 建议英文名 |
|-----------|-----------|
| `tools/感受野.py` | `tools/receptive_field.py` |
| `tools/感受野可视化.py` | `tools/receptive_field_vis.py` |
| `tools/批量置空最终模型文件.py` | `tools/batch_zero_models.py` |
| `tools/拼接感受野图.py` | `tools/stitch_rf_images.py` |
| `tools/注意力可视化.py` | `tools/attention_vis.py` |
| `tools/深度图单通道转三通道.py` | `tools/depth1c_to_3c.py` |
| `tools/域解释.py` | `tools/domain_interpret.py` |
| `strategies/策略方案汇总.md` | `strategies/strategy_summary.md` |
| `strategies/新建有用策略.md` | `strategies/new_strategies.md` |

**风险：** Windows 文件系统、CI/CD 管道、部分 git 工具对非 ASCII 文件名支持不佳。

### 2.5 缺少打包配置

无 `setup.py` / `pyproject.toml`，无法通过 `pip install -e .` 安装。

**建议：** 添加 `pyproject.toml`：
```toml
[project]
name = "ssl4mis"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [...]  # 从 requirements.txt 迁移

[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

---

## 三、建议修复（代码质量）

### 3.1 推理未设置随机种子

`core/test.py` 的 `main()` 没有调用 `setup_seed()`。若模型含 Dropout 等随机操作，推理结果不可复现。

### 3.2 FeCLoss 设备反模式

`utils/losses.py:152` — `FeCLoss.__init__` 存储 `self.device`，`forward()` 中用它创建 tensor。模块 `.to(device)` 后 `self.device` 不更新，导致设备不匹配。

```python
# 当前（有问题）
self.device = device
identity = torch.eye(N, device=self.device)

# 修复：从输入 tensor 推断设备
identity = torch.eye(N, device=z_i.device)
```

### 3.3 代码重复

`strategies/semi_mt_depth_guider_proto_teacher_v3.py:110` 的 `_create_branch_model` 与 `base_strategy.py` 的 `_create_ema_model` 几乎完全相同（基于 `inspect.signature` 重建模型）。建议提取为基类可复用方法。

### 3.4 复杂度过高

`strategies/semi_mt_depth_guider_proto_teacher_v3.py`：
- `compute_loss` 约 150 行，混合 5 种 loss 计算
- `training_step` 中 AMP/非 AMP 路径的梯度裁剪逻辑重复 6 次

**建议：** 拆分为子方法，optimizer 处理用循环。

### 3.5 全局副作用

| 位置 | 问题 |
|------|------|
| `core/train.py:28` | 模块级 `warnings.filterwarnings("ignore")` 影响整个进程 |
| `core/test.py:35` | 同上 |
| `core/train.py:466` | `setup_logging` 删除所有 root handler |

**建议：** 用 `logging.getLogger(__name__)` 替代 root logger；warnings 过滤移入函数内部。

### 3.6 print 语句应改用 logging

| 文件 | 行号 | 内容 |
|------|------|------|
| `models/networks/depth.py` | 92 | 预训练权重加载信息 |
| `models/networks/dformerv2_small.py` | 547 | 预训练权重加载信息 |
| `models/networks/block/resnetunet_block.py` | 67 | 预训练权重加载信息 |
| `strategies/semi_proto_v1.py` | 65-66 | prototype 初始化信息 |

### 3.7 未使用的导入

`core/train.py:6` — `field` 从 `dataclasses` 导入但未使用。

---

## 四、建议改进（测试与文档）

### 4.1 缺失的测试覆盖

| 模块 | 现状 | 建议 |
|------|------|------|
| `utils/losses.py` | 无专门测试 | 添加 `test_losses.py` |
| `utils/metrics.py` | 无专门测试 | 添加 `test_metrics.py` |
| `semi_mt_depth_guider_proto_teacher_v3` | 无测试 | 最复杂策略（5 个 optimizer），应有单元测试 |
| `core/test.py::calculate_metric_percase` | 仅间接测试 | 添加 numpy 层面单元测试 |

### 4.2 requirements.txt 补充

缺少 `pytest`（测试中使用但未列入）。

### 4.3 .gitignore 补充

当前缺少：
```
*.csv
log.txt
CODE_REVIEW_REPORT.md
CODE_QUALITY_REVIEW.md
.claude/
```

---

## 五、架构亮点

| 设计 | 说明 |
|------|------|
| 策略注册表 `STRATEGY_SPECS` | 集中管理 28 种策略元数据，新增策略只需 3 步 |
| 模型工厂 `MODEL_REGISTRY` | 50+ 模型变体统一注册，按策略名自动解析 |
| 动态参数注入 `StrategyArgumentParser` | 只注入当前策略的 CLI 参数，避免参数冲突 |
| GroupNorm 替代 BatchNorm | 解决半监督训练中 BN 统计量不稳定问题 |
| EMA 教师模型 | `base_strategy.py` 统一管理，子策略无需重复实现 |
| 完善的随机种子设置 | `setup_seed` + worker_init_fn 保证训练可复现 |

---

## 六、开源前 Checklist

- [ ] 添加 LICENSE 文件
- [ ] 编写英文 README（保留中文版为 `README_CN.md`）
- [ ] 清理 `tools/` 中所有硬编码绝对路径
- [ ] 中文文件名改为英文
- [ ] 添加 `pyproject.toml`
- [ ] `core/test.py` 入口设置随机种子
- [ ] 修复 `FeCLoss` 设备反模式
- [ ] `print` 改 `logging`
- [ ] 删除未使用的 `field` 导入
- [ ] 补充 `.gitignore` 规则
- [ ] 添加 `utils/losses.py` 和 `utils/metrics.py` 单元测试
- [ ] 重命名 `strategies/` 下中文 markdown 文档

---

## 七、总结

代码整体质量**良好**。架构清晰（策略注册表 + 模型工厂），核心训练/测试路径无硬编码，测试覆盖合理。主要问题：

1. **可移植性**：tools/ 和 tests/ 的硬编码路径 + 中文文件名
2. **开源规范**：缺 LICENSE、英文 README、打包配置
3. **代码细节**：FeCLoss 设备 bug、test.py 缺种子、代码重复

修复工作量不大，预计 1-2 天可完成开源准备。
