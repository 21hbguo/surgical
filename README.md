# SSL4MIS Code Navigation

医学图像分割训练框架，当前主线是腹腔镜手术器械分割、多任务标签、半监督策略、RGB/RGBD 输入、Mean Teacher 系列策略复现与改造。

## 1. 快速入口

```bash
python -m core.train --task 1 --way mt --exp endovis2017/MT --labeled_num 10 --fold 0
python -m core.test --task 1 --way mt --exp endovis2017/MT --labeled_num 10 --fold 0 --lr 3e-5 --pth_type best
```

核心参数：

- `--task {1,2,3}`：选择 `task{n}.json` 和对应标签目录。
- `--way`：策略名，真实列表来自 `strategies/specs.py::STRATEGY_SPECS`。
- `--exp DATASET/EXP_NAME`：自动推断数据根目录为 `../data/DATASET`。
- `--model`：为空时由 `--way` 和 `--pretrain` 自动解析。
- `--pretrain {none,resnet,depth,dinov3}`：控制默认模型前缀。
- `--use_depth {1,3,13}`：启用深度模态；`13` 表示同时加载 depth1c 和 depth3c，但模型输入按 depth1c 处理。
- `--normalize {255,minmax,imagenet}`：图像/深度当前共用该归一化模式。
- `--optimizer adam`：当前训练实现只构建 Adam。
- `--fold -1`：训练/测试全部 folds。

## 2. AI 导航地图

- `core/args.py`：训练/测试 CLI 参数、动态策略私有参数注入、运行参数 finalize、默认模型解析入口。
- `core/train.py`：训练主入口、DataLoader 构造、模型/优化器/策略实例化、训练循环、验证、checkpoint 保存。
- `core/test.py`：测试主入口、checkpoint 查找、推理、指标统计、可视化导出。
- `core/runtime.py`：设备、fold、输出路径、checkpoint 路径等运行期解析。
- `strategies/specs.py`：策略注册中心，维护 `STRATEGY_SPECS`、半监督标记、默认模型后缀、输入通道解析、策略私有参数注入。
- `strategies/base_strategy.py`：策略基类，包含通用训练辅助、EMA、噪声扰动、验证默认逻辑。
- `strategies/*.py`：具体策略实现；策略特有参数写在本文件的 `@staticmethod add_args(parser)`，不要从其他策略版本 import 私有 args。
- `models/factory.py`：模型注册中心，维护 `MODEL_REGISTRY`、模型构造参数映射、默认模型名解析。
- `models/networks/`：UNet、ResNetUNet、DepthUNet、DINOv3UNet、DFormerV2、Ternaus 等网络实现。
- `data/dataset.py`：数据读取、RGB/depth/label 路径解析、归一化、task 标签选择。
- `data/transforms.py`：训练/验证 transform。
- `data/samplers.py`：有标注/无标注 batch sampler。
- `utils/losses.py`：监督、半监督、一致性、对比学习等 loss。
- `utils/metrics.py`：Dice、IoU、Precision、Recall、Acc、depth 质量指标。
- `utils/lr_scheduler.py`：学习率调度。
- `tests/`：参数解析、策略注册、模型工厂、训练入口、策略行为、归一化、测试指标等回归测试。
- `scripts/`：常用训练/测试 shell 脚本。

## 3. 当前参数机制

参数分三层：

- 公共参数：写在 `core/args.py::add_common_args`。
- 训练参数：写在 `core/args.py::add_train_args`。
- 测试参数：写在 `core/args.py::add_test_args`。
- 策略私有参数：写在对应策略文件的 `add_args(parser)`。

解析流程：

1. `StrategyArgumentParser` 先从 argv 读取 `--way`。
2. 根据 `--way` 调用 `strategies/specs.py::add_strategy_args`。
3. 只把当前策略的私有参数加入 parser。
4. 非当前策略的私有参数会被拒绝。

因此，新增策略参数不要写进 `core/args.py`，也不要写进 `parser.set_defaults(...)`；应直接写进对应策略类的 `add_args(parser)`。

## 4. 策略注册速查

策略真实来源：

```python
strategies/specs.py::STRATEGY_SPECS
```

当前主要策略族：

- 全监督：`fully`、`fully_reg`、`ternaus`、`dformerv2_fully`。
- 全监督增强/预训练：`fully_rgb_masking_depth_v1`、`fully_depth_pretrain_v1`、`fully_depth_pretrain`、`fully_contrast_v1`、`fully_contrast_v1_1`。
- Mean Teacher：`mt`、`uamt`、`depth_mt`。
- 深度引导 MT：`mt_depth_teacher_v1`、`mt_depth_guider_v1`、`mt_depth_guider_v2`、`mt_depth_guider_v3`。
- 深度原型 MT：`mt_depth_guider_proto_v1`、`mt_depth_guider_proto_teacher_v1`、`mt_depth_guider_proto_teacher_v2`、`mt_depth_guider_proto_teacher_v3`。
- 其他半监督：`proto`、`proto_v1`、`dycon`、`w2s`、`urpc`、`rdnet`、`semi_mean_teacher_contrast_v1`、`semi_mean_teacher_text_v1`、`only_depth_input`。

模型默认解析规则：

- `fixed_model_name` 非空时直接使用固定模型。
- 否则由 `--pretrain` 得到前缀：`none -> unet`，`resnet -> resnet`，`depth -> depth`，`dinov3 -> dinov3`。
- 再拼接策略的 `model_suffix`。
- 最终模型名必须存在于 `models/factory.py::MODEL_REGISTRY`。

## 5. 数据结构

`--exp endovis2017/MT` 会推断数据根目录为 `../data/endovis2017`。

```text
data_root/
├── train_slices.list
├── val_slices.list
├── test_slices.list
├── train_slices_f0.list
├── val_slices_f0.list
├── task1.json
├── task2.json
├── task3.json
└── data/
    ├── images/
    ├── labels_task1_binary/
    ├── labels_task2_part/
    ├── labels_task3_class/
    ├── depth1c_slices/
    └── depth3c_slices/
```

`task{n}.json` 必需字段：

```json
{
  "num_classes": 2,
  "n_folds": 4,
  "input_channels": 3,
  "classes": [
    {"name": "Background", "label_id": 0, "color": [0, 0, 0]},
    {"name": "Instrument", "label_id": 1, "color": [255, 0, 0]}
  ]
}
```

`num_classes`、`n_folds`、`input_channels` 只从当前 task json 读取；缺字段会报错。

## 6. 输出结构

训练默认输出：

```text
../result_train/{dataset}_Sampling{sampling}/task{task}/{exp_name}/{labeled_num}_labeled_lr{lr}_{model_name}/[f{fold}]/
├── log.txt
├── data_train_labeled.list
├── data_train_unlabeled.list
├── data_val.list
├── model_best.pth
├── model_final.pth
└── visualizations/
```

测试默认输出：

```text
../result_predict/{dataset}_Sampling{sampling}/task{task}/{exp_name}/{labeled_num}_labeled_lr{lr}_{model_name}/[f{fold}]/
```

测试 checkpoint 从 `--train_result_root` 读取，默认是 `../result_train`；测试时 `--lr`、`--model`、`--pretrain`、`--use_depth` 必须与训练路径匹配。

## 7. 新增策略流程

1. 在 `strategies/` 新建策略文件并实现策略类。
2. 策略类继承 `BaseTrainingStrategy` 或兼容其接口。
3. 策略私有 CLI 参数写成 `@staticmethod add_args(parser)`。
4. 在 `strategies/specs.py` import 策略类。
5. 在 `STRATEGY_SPECS` 添加 `_spec(...)`，明确 `is_semi`、`model_suffix`、`fixed_model_name`、`in_chns`。
6. 如需新模型，在 `models/factory.py::MODEL_REGISTRY` 注册模型名和参数映射。
7. 如需对外 import，在 `strategies/__init__.py` 补充导出。
8. 添加或更新 `tests/` 中对应解析、注册、模型工厂和策略行为测试。

最小 `add_args` 示例：

```python
@staticmethod
def add_args(parser):
    parser.add_argument("--proto_feature_dim", type=int, default=64)
    parser.add_argument("--proto_weight", type=float, default=0.1)
```

## 8. 常用验证命令

```bash
python -m pytest tests/test_args_runtime.py tests/test_train_entrypoint.py -q
python -m pytest tests/test_strategy_specs.py tests/test_model_factory.py -q
python -m pytest tests/test_normalize_modes.py tests/test_runtime_paths.py -q
```

## 9. 维护约定

- `README.md` 用于快速导航；实现细节以源码和测试为准。
- `strategies/specs.py` 是策略注册唯一权威来源。
- `models/factory.py` 是模型注册唯一权威来源。
- 策略私有参数必须独立写在对应策略文件内。
- 训练入口和测试入口统一使用 `python -m core.train`、`python -m core.test`。
- 半监督策略至少需要 1 个 labeled 样本和 1 个 unlabeled 样本。
- `mt` 及其变体验证/测试阶段默认使用 student 分支推理。
