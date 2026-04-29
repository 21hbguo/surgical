# 新增策略指南

新增一个策略（新模型 + 旧backbone）的标准流程。

---

## 文件改动清单

### 1. 新建策略文件

**`strategies/semi_xxx.py`**

```python
from strategies.base_strategy import BaseTrainingStrategy
# 或继承已有的 MeanTeacherStrategy 等

class YourStrategy(BaseTrainingStrategy):
    def __init__(self, args, model, optimizer, device):
        super().__init__(args, model, optimizer, device)
        # 如需 EMA teacher
        # self._enable_ema_support()
        # 如有额外子模块/optimizer，在此初始化

    def compute_loss(self, batch_data, iter_num, epoch):
        # 核心训练逻辑
        # 返回 dict: {"loss": total_loss, "loss_detail": ...}
        pass

    # 可选：重写以下方法
    # def training_step(self, batch_data, iter_num, epoch): ...
    # def validation_step(self, batch_data): ...
    # def get_state_dict(self): ...
    # def load_state_dict(self, state_dict): ...
```

---

### 2. 注册策略（2个文件）

**`strategies/__init__.py`** — 添加 import

```python
from strategies.semi_xxx import YourStrategy
```

**`strategies/specs.py`** — 在 `STRATEGY_SPECS` 添加条目

```python
"your_way": _spec("your_way", YourStrategy, is_semi=True, model_suffix="your_suffix"),
```

参数说明：

| 参数 | 作用 |
|---|---|
| `name` | CLI `--way` 参数名 |
| `cls` | 策略类 |
| `is_semi` | 半监督=True, 全监督=False |
| `model_suffix` | 模型名后缀，最终模型名 = `{pretrain_prefix}_{suffix}` |
| `fixed_model_name` | 直接指定模型名（覆盖 suffix 拼接逻辑） |
| `in_chns` | 输入通道数，`"metadata"` 表示从 task.json 读取 |

模型名解析示例：
- `--way your_way --pretrain none` → `unet_your_suffix`
- `--way your_way --pretrain resnet` → `resnet_your_suffix`
- `--way your_way --pretrain dinov3` → `dinov3_your_suffix`

---

### 3. 新增网络变体类

在对应 backbone 文件中新增类，复用已有 encoder，改 decoder head 或添加新分支：

| Backbone | 文件 | 基类参考 |
|---|---|---|
| UNet | `models/networks/unet.py` | `UNet_Base`, `UNet_ContrastV1` 等 |
| ResNet | `models/networks/resnet.py` | `ResNet_Base`, `ResNet_ContrastV1` 等 |
| Depth | `models/networks/depth.py` | `Depth_Base` 等 |
| DINOv3 | `models/networks/dinov3.py` | `DINOv3_Base` 等 |

输出格式约定：

| 返回类型 | 用途 | 示例策略 |
|---|---|---|
| `seg_logits` (单tensor) | 标准分割 | fully, mt |
| `(seg_logits, feat)` (tuple) | 带额外特征 | contrast_v1, proto |
| `(dp0, dp2, dp3, dp4)` (tuple) | 多尺度监督 | urpc |
| `(main, sub1, sub2, sub3)` (tuple) | 多解码器 | w2s |
| `(seg_logits, depth_output)` (tuple) | 联合深度预测 | depth |

**关键**：所有 UNet 系模型需存储 `self.params` 字典（用于 EMA 重建）。

如引入全新子模块，在 `models/networks/` 下新建文件（参考 `prototype.py`）。

---

### 4. 注册模型

**`models/factory.py`** — 在 `MODEL_REGISTRY` 添加条目

按需为每个 backbone 前缀添加：

```python
"unet_your_suffix":   ModelSpec(builder=UNet_YourVariant,   arg_map={"in_chns": "in_chns", "feature_scale": "feature_scale"}),
"resnet_your_suffix": ModelSpec(builder=ResNet_YourVariant, arg_map={"in_chns": "in_chns", "feature_scale": "feature_scale"}),
# depth_, dinov3_ 按需添加
```

`arg_map` 将构造函数参数映射到 `args` 属性名。

---

### 5. 其他可能需要修改的文件

| 文件 | 何时改 | 改什么 |
|---|---|---|
| `utils/losses.py` | 新增 loss 类型 | 添加新 loss 类 |
| `core/test.py` | 模型输出格式不同 | `_predict_logits()` 加分支处理 |
| `core/args.py` | 策略有特殊 CLI 参数 | 添加 argparse 参数 |
| `data/dataset.py` | 需加载额外数据 | 扩展 `__getitem__` |
| `scripts/` | 跑实验 | 添加 train/test 脚本 |

---

## 最小改动路径（5个文件）

```
strategies/semi_xxx.py        ← 新建策略类
strategies/__init__.py        ← 加 import
strategies/specs.py           ← 注册 StrategySpec
models/networks/unet.py       ← 新增网络变体（或其他 backbone 文件）
models/factory.py             ← 注册 ModelSpec
```

---

## 自检清单

- [ ] 策略类实现了 `compute_loss()` 且返回 `{"loss": ...}`
- [ ] `STRATEGY_SPECS` 中 `model_suffix` 与 `MODEL_REGISTRY` 中的 key 后缀一致
- [ ] 网络变体类存储了 `self.params`（用于 EMA 重建）
- [ ] 模型输出格式与策略 `compute_loss()` 解包方式匹配
- [ ] 如输出格式特殊，`core/test.py` 的 `_predict_logits()` 已处理
- [ ] `--pretrain` 各选项（none/resnet/depth/dinov3）对应的模型 key 都已在 registry 中
