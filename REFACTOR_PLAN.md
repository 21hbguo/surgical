# 项目重构建议

## 一、当前问题诊断

### 1. 根目录杂乱（22 个条目，非代码文件占比过高）

| 类型 | 文件 | 建议 |
|------|------|------|
| 研究计划 | `深度扰动研究计划_v1/v2_20260527.md` | 移至 `docs/` |
| 分析文档 | `georisk_spc_unetbackbone_analysis.md` | 移至 `docs/` |
| 计划/待办 | `EXPERIMENT_TODO.md`, `way_exp.md`, `TRAINING_FRAMEWORK_FIX_PLAN.md`, `SESSION_BACKUP.md` | 合并为 `docs/TODO.md` 或移除 |
| 论文产物 | `paper/`, `paper copy/`, `paper_lab/` | 保留 `paper/`（当前论文），`paper copy/` 删除，`paper_lab/` → `tools/` |
| 日志 | `main.log`, `texput.log` | `.gitignore` 或清理 |
| 空文件 | `test.py` | 删除（入口在 `core/test.py`） |
| Shell 脚本 | `run_ablation.sh` ×4 + `run_baselines.sh` ×5 | 合并为 `scripts/run.sh` 参数化 |
| CSV | `ablation_tracking.csv`, `experiment_tracking.csv` | 移至 `results/` |

### 2. strategies/ 严重膨胀（33 文件, 6632 行）

核心问题：
- **v1/v2/v3 序列重复**：`mt_depth_guider_v1/v2/v3`、`mt_depth_guider_proto_teacher_v1/v2/v3` 等大文件（260-450 行/个）之间差异极小，仅 loss 组合或权重不同
- **`__init__.py` 手动维护**：36 行纯 import 列表，新增策略需改两处（文件 + `__all__`）
- **`specs.py` 职责过载**：注册表定义 + 策略别名 + channel 解析 + args 注入，201 行
- **`base_strategy.py` 过重**（172 行）：EMA 创建含 inspect 反射、参数别名解析等，应拆分

### 3. tests/ 过度测试（23 文件, 3479 行）

- `test_task_dataset_selection.py` 单文件 880 行
- `test_args_runtime.py` 299 行
- `test_test_metrics.py` 284 行
- 多个策略单独测试文件可合并

### 4. tools/ 命名和结构混乱

- 中文文件名：`感受野.py`、`批量置空最终模型文件.py`、`拼接感受野图.py`、`域解释.py`、`深度图单通道转三通道.py`、`注意力可视化.py`
- 功能重叠：`diffSLIC.py` + `diffslic_benchmark.py`
- 工具脚本散落 20+ 个文件

### 5. 违反 AGENTS.md 规则

| 规则 | 现状 |
|------|------|
| "禁止为只调用 1 次的代码定义函数/类" | `core/testing/export.py`、`core/testing/visualization.py` 中大量单次使用封装 |
| "策略文件只写核心差异" | 多个 semi strategy 重复写 dataloader 逻辑、EMA 更新 |
| "删除冗余优先于新增包装" | `core/test_gamma_beta.py`（23070 行）是实验产物，不应在 core/ |
| "参数只解析 1 次" | `specs.py` 中 `resolve_strategy_input_settings` 和 `args.py` 中重复推断 `in_chns` |
| "不生成注释" | 多处策略文件顶部存在 docstring 注释块 |

---

## 二、重构方案（不改变功能，极大简化）

### Phase 1: 清理根目录

```
# 新建目录
mkdir -p docs results scripts

# 迁移文档
mv 深度扰动研究计划_*.md georisk_spc_unetbackbone_analysis.md docs/
mv EXPERIMENT_TODO.md way_exp.md TRAINING_FRAMEWORK_FIX_PLAN.md SESSION_BACKUP.md docs/

# 删除冗余
rm -rf "paper copy/" __pycache__ .pytest_cache test.py main.log texput.log

# 迁移 CSV
mv ablation_tracking.csv experiment_tracking.csv results/

# 删除 run_*.sh，迁移到 scripts/
mkdir -p scripts
mv run_*.sh scripts/
```

### Phase 2: 合并 shell 脚本

9 个 `run_*.sh` 合并为 `scripts/run.sh`：
- 通过 `--mode ablation|baseline|task2_task3` 区分
- 通过 `--config` 引用 configs（YAML 或 JSON）
- 消除重复参数（batch size、lr、max_iterations 等）

### Phase 3: 精简 strategies/

#### 3a. 自动注册机制

```python
# strategies/__init__.py — 替换当前 70 行手动 import
import importlib, pkgutil, sys
from .specs import STRATEGY_REGISTRY, create_strategy, get_strategy_names

_strategy_dir = __path__[0]
for _, name, _ in pkgutil.iter_modules([_strategy_dir]):
    if name.startswith("_") or name in ("specs", "base_strategy"):
        continue
    module = importlib.import_module(f".{name}", __name__)
    for attr in dir(module):
        if attr.endswith("Strategy") and attr != "BaseTrainingStrategy":
            globals()[attr] = getattr(module, attr)

__all__ = ["STRATEGY_REGISTRY", "create_strategy", "get_strategy_names"]
# plus all strategy classes discovered dynamically
```

#### 3b. 合并 v1/v2/v3 系列

将 `mt_depth_guider_proto_teacher_v1/v2/v3`（合计 ~1050 行）合并为单个策略类，通过 args 参数控制差异行为：

```python
# semi_mt_depth_guider_proto_teacher.py (~300 行, 替代 1050 行)
# --proto_version v1|v2|v3 控制：
#   v1: student uses proto + CE+Dice; depth_teacher trained on labeled
#   v2: student uses CE+Dice only; depth_teacher on unlabeled consistency
#   v3: unified consistency; shared proto between branches
```

同理处理 `mt_depth_guider_v1/v2/v3`（~210 行 → ~120 行）。

预期节省：~800 行

#### 3c. 拆分 base_strategy.py

```
strategies/
├── base_strategy.py        # ~60 行: 抽象接口 + training_step + backward
├── ema_mixin.py            # ~80 行: _create_ema_model, _update_ema, _enable_ema_support
├── noise_mixin.py          # ~20 行: _add_noise
├── consistency_mixin.py    # ~15 行: _get_consistency_weight
├── specs.py                # ~120 行: 注册表 + create_strategy + resolve
└── <strategy files>
```

### Phase 4: 精简 tests/

- 合并策略测试：`test_fully_contrast_v1.py` + `test_fully_contrast_v1_1.py` → `test_fully_contrast.py`
- 合并 mt_depth_guider 系列测试 → `test_mt_depth_guider.py`
- `test_task_dataset_selection.py`（880 行）拆为 `test_dataset_selection.py` + `test_data_loading.py`
- 删除或合并极小测试文件（<30 行）：`test_unet_contrast_v1.py`、`test_resnet_contrast_v1.py`、`test_normalize_modes.py`

预期：23 文件 → 10-12 文件，3479 行 → ~2000 行

### Phase 5: 清理 core/

- `core/test_gamma_beta.py`（23070 行）→ 移至 `tools/` 或删除（实验产物）
- `core/testing/export.py`（9223 行）→ 拆分为 `export_csv.py` + `export_results.py`
- `core/testing/visualization.py`（8742 行）→ 拆分为 `gradcam.py` + `confidence_viz.py` + `rgb_viz.py`
- `core/train_h5.py` → 合并到 `core/train.py`（通过 `--data-format png|h5` 参数）

### Phase 6: 清理 tools/

中文文件名重命名为英文：
| 原名 | 新名 |
|------|------|
| `感受野.py` | `receptive_field.py` |
| `感受野可视化.py` | `receptive_field_viz.py` |
| `拼接感受野图.py` | `stitch_receptive_field.py` |
| `批量置空最终模型文件.py` | `batch_clear_models.py` |
| `深度图单通道转三通道.py` | `depth_1to3_channel.py` |
| `注意力可视化.py` | `attention_viz.py` |
| `域解释.py` | `domain_explain.py` |

合并功能重叠文件，删除无用脚本。

### Phase 7: 迁移 paper_lab/ 内容

```
paper_lab/compute_boundary_metrics.py  → tools/
paper_lab/gen_convergence.py           → tools/
paper_lab/aggregate_task2_task3.py     → tools/
paper_lab/generate_paper_tables.py     → tools/
paper_lab/statistical_tests.py         → tools/
paper_lab/visualize_*.py               → tools/viz/
```

然后删除 `paper_lab/` 目录。

---

## 三、预期效果

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| 根目录条目 | 22 | ~12 | -45% |
| strategies/ 文件 | 33 | ~25 | -24% |
| strategies/ 行数 | 6632 | ~4500 | -32% |
| tests/ 文件 | 23 | ~12 | -48% |
| tests/ 行数 | 3479 | ~2000 | -42% |
| tools/ 中文文件 | 7 | 0 | 100% |
| 独立目录 | paper/, paper copy/, paper_lab/ | paper/, tools/ | 合并 |
| shell 脚本 | 9 | 1-2 | -80% |

---

## 四、执行顺序建议

1. **Phase 1**（根目录清理）— 零风险，立即执行
2. **Phase 6**（tools 重命名）— 纯重命名，不影响功能
3. **Phase 7**（paper_lab 迁移）— 修改 import 路径即可
4. **Phase 5**（core 清理）— test_gamma_beta.py 移除
5. **Phase 3**（strategies 精简）— 核心简化，需要测试验证
6. **Phase 2**（shell 脚本合并）— 功能等价替换
7. **Phase 4**（tests 精简）— 最后做，确保不影响 CI

每步完成后运行 `pytest tests/ -q` 验证。
