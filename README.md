# SSL4MIS: Semi-Supervised Learning for Medical Image Segmentation

统一的医学图像分割训练框架，支持多种半监督策略（单一训练/测试入口，便于复现实验和横向对比）。

## 1. 当前版本说明

本仓库当前有效入口与参数如下：

- 训练入口：`python -m core.train`
- 测试入口：`python -m core.test`
- 策略参数：`--way`（不是 `--strategy`）
- 参数定义与默认值：`core/args.py`
- 策略注册：`strategies/__init__.py`

如果你之前用的是旧命令（如 `python train.py --strategy ...`），请改为本 README 的命令格式。

## 2. Features

- 多策略统一接口：`fully`, `mt`, `uamt`, `urpc`, `ternaus`, `proto`, `dycon`, `w2s`, `depth_mt`, `dformerv2_fully`
- 模型工厂统一创建：`models/factory.py`
- 支持 RGB / RGBD（`--use_depth 1|3`）
- 支持多折训练与测试（`--fold`，可用 `-1` 跑全部 folds）
- 训练自动保存 best/final checkpoint 与数据划分清单

## 3. Installation

```bash
git clone <your-repo-url>
cd code_all
pip install -r requirements.txt
```

建议 Python 3.10 + CUDA 环境。

## 4. Data Format

数据集目录（由 `--exp` 的 `/` 前缀自动解析得到 `../data/{dataset}`）建议组织为：

```text
data_root/
├── train_slices.list
├── val_slices.list
├── test_slices.list
├── train_slices_f0.list        # 可选：多折
├── val_slices_f0.list          # 可选：多折
├── task1.json                  # 必需：task1 元信息
├── task2.json                  # 必需：task2 元信息
├── task3.json                  # 必需：task3 元信息
└── data/
    ├── images/
    │   ├── xxx.png
    │   └── ...
    ├── labels_task1_binary/
    ├── labels_task2_part/
    ├── labels_task3_class/
    ├── depth1c_slices/         # 可选：1通道深度
    └── depth3c_slices/         # 可选：3通道深度
```

### `task{n}.json`（必需）

以下字段是必填项，且是唯一来源：

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

说明：

- `num_classes`、`n_folds`、`input_channels` 必须写在对应的 `task{n}.json` 中。
- 训练和测试不会再从路径名、CLI 默认值或配置默认值回退推断这些字段。
- 缺少对应的 `task{n}.json` 或缺少上述字段时，程序会直接报错。
- 标签目录按任务自动解析，约定为 `data/labels_task{n}_*`。
- `train/val/test` 列表里的样本名既可以写 `case_a`，也可以写 `case_a.png`；测试集会保留 `patient.001` 这种带点号的 stem。

## 5. Quick Start

以下命令均在仓库根目录执行。

### 5.1 全监督训练

```bash
python -m core.train \
  --way fully \
  --exp endovis2017/Fully \
  --labeled_num 10 \
  --fold 0
```

### 5.2 半监督训练（Mean Teacher）

```bash
python -m core.train \
  --way mt \
  --exp endovis2017/MT \
  --labeled_num 10 \
  --fold 0 \
  --lr 1e-4
```

### 5.3 深度输入训练（例如 Proto）

```bash
python -m core.train \
  --way proto \
  --exp endovis2017/Proto \
  --labeled_num 10 \
  --fold 0 \
  --use_depth 3
```

### 5.4 测试

```bash
python -m core.test \
  --way mt \
  --exp endovis2017/MT \
  --labeled_num 10 \
  --fold 0 \
  --lr 1e-4 \
  --pth_type best
```

注意：`python -m core.test` 默认会到训练结果根目录 `../result_train` 下，
根据 `exp/labeled_num/lr/model` 组合路径找 checkpoint，
测试时 `--lr` 需要与训练保持一致，否则会报 checkpoint not found。
默认参数是：

- `python -m core.train` 默认 `--result_root ../result_train`
- `python -m core.test` 默认 `--result_root ../result_predict`
- `python -m core.test` 默认 `--train_result_root ../result_train`
- 所有相对路径参数，例如 `--snapshot_path ../xxx` 或 `--result_root ../xxx`，都会按命令执行时的当前工作目录解析
- `mt` 及其变种在验证/测试阶段统一使用 student 模型做推理；teacher 分支仅参与训练

### 5.5 跑全部 folds

```bash
python -m core.train --way mt --exp endovis2017/MT --labeled_num 10 --fold -1
python -m core.test  --way mt --exp endovis2017/MT --labeled_num 10 --fold -1 --lr 1e-4
```

`--fold -1` 只依赖对应 `task{n}.json` 的 `n_folds`。

## 6. Supported Strategies

| `--way` | 类型 | 默认模型 (`core/args.py`) |
|---|---|---|
| `fully` | supervised | `unet` |
| `mt` | semi-supervised | `unet` |
| `uamt` | semi-supervised | `unet` |
| `urpc` | semi-supervised | `unet_urpc` |
| `ternaus` | supervised | `ternaus16` |
| `proto_v1` | semi-supervised | `unet_proto_v1` |
| `proto` | semi-supervised | `unet_proto_v1` |
| `dycon` | semi-supervised | `unet_dycon` |
| `w2s` | semi-supervised | `unet_w2s` |
| `depth_mt` | semi-supervised | `unet_depth` |
| `dformerv2_fully` | supervised | `dformerv2_small` |

## 7. Common Arguments

训练常用参数（`python -m core.train`）：

```bash
--way {fully,mt,uamt,urpc,ternaus,proto_v1,proto,dycon,w2s,depth_mt}
--exp NAME
--model MODEL_TYPE
--labeled_num FLOAT
--labeled_bs INT
--unlabeled_bs INT
--sampling {none,interval}
--max_iterations INT
--lr FLOAT
--fold INT
--resize_size H W
--use_depth {1,3}
--normalize {minmax,255,imagenet}
--val_iter INT
--debug
```

测试常用参数（`python -m core.test`）：

```bash
--way ...
--exp NAME
--labeled_num FLOAT
--sampling {none,interval}
--fold INT
--lr FLOAT
--pth_type {best,final,latest}
--use_depth {1,3}
--batch_size INT
--snapshot_path PATH
```

当前优化器固定为 `Adam`。

`--labeled_num` 语义：

- 始终按百分数解释，不再区分“比例模式”和“百分数模式”
- `0.1` 表示 `0.1%`
- `1` 表示 `1%`
- `10` 表示 `10%`
- `40` 表示 `40%`
- 当训练切片列表非空且 `labeled_num > 0` 时，框架会至少保留 1 个 labeled slice

## 8. Output Structure

默认训练输出目录（由 `python -m core.train` 自动构造）：

```text
../result_train/{dataset}_Sampling{sampling}/task{task}/{exp_name}/{labeled_num}_labeled_lr{lr}_{model_name}/[f{fold}]/
├── log.txt
├── data_train_labeled.list          # 半监督
├── data_train_unlabeled.list        # 半监督
├── data_val.list
├── model_best.pth
├── model_final.pth
└── visualizations/
    └── latest.png
```

其中 `dataset` 段会自动带上 normalize 和 sampling 后缀，例如 `endovis2017_255_Samplingnone`。

测试会在对应目录写入：

- `test_results.csv`
- `fold_summary_best.csv` / `fold_summary_final.csv`
- 聚合结果（`../result_predict/{dataset}_Sampling{sampling}/...`）

## 9. Project Structure

```text
code_all/
├── core/
│   ├── args.py
│   ├── train.py
│   └── test.py
├── strategies/
├── models/
├── data/
├── utils/
├── scripts/
└── README.md
```

## 10. Batch Scripts

可直接参考：

- `scripts/train_17.sh`
- `scripts/test_17.sh`
- `scripts/train_kvasir_all.sh`
- `scripts/test_kvasir.sh`

## 11. Extending

### 新增策略

1. 新建 `strategies/xxx.py`，继承 `BaseTrainingStrategy`。
2. 在 `strategies/__init__.py` 注册到 `STRATEGY_REGISTRY`。
3. 在 `core/args.py` 的策略映射和 `SEMI_STRATEGIES` 中补充。

### 新增模型

1. 在 `models/networks/` 添加网络实现。
2. 在 `models/factory.py` 中新增分支。
3. 通过 `--model` 或策略映射调用。

## 12. License

MIT License，见 `LICENSE`。

## 13. Contact

如有问题可提 issue 或联系：`2678896985@qq.com`
