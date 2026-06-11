# 训练框架修复方案
## 结论
当前框架属于研究实验可用状态，未达到稳定训练平台标准。修复优先级按“先恢复测试一致性，再补训练可靠性，再清理默认行为”执行。
## P0：恢复测试绿灯
### 现状
`python -m pytest -q` 当前结果：`141 passed, 14 failed`。
失败集中在默认学习率、默认模型、路径命名、EMA 模式、Trainer 参数兼容。
### 修复项
1. 对齐默认学习率
目标文件：`core/args.py`
修复：将 `--lr` 默认值恢复为测试与 README 期望的 `3e-5`，或同步修改测试与 README 为 `1e-4`。建议恢复为 `3e-5`，因为现有测试和快速入口均指向该值。
验收：`tests/test_args_runtime.py::ArgsRuntimeTest::test_train_parser_uses_config_defaults_as_args_defaults` 通过。
2. 对齐默认模型解析
目标文件：`core/args.py`、`strategies/specs.py`
修复：明确 `--pretrain` 默认语义。当前代码默认 `none -> unet`，测试期望默认模型为 `resnet`。建议把 `--pretrain` 默认值改为 `resnet`，并保留显式 `--pretrain none` 时解析到 `unet`。
验收：默认 `fully/mt/proto/fully_contrast_v1` 路径中的模型名为 `resnet` 系列；显式 `--pretrain none` 仍为 `unet` 系列。
3. 对齐测试可视化默认值
目标文件：`core/args.py`
修复：将测试参数 `--rgb` 默认值从 `0` 改为 `2`，或同步修改测试。建议改为 `2`，因为测试覆盖其作为默认导出行为。
验收：`tests/test_args_runtime.py::ArgsRuntimeTest::test_test_parser_defaults_include_relative_result_roots` 通过。
4. 修复 EMA train/eval 行为
目标文件：`strategies/base_strategy.py`
修复：`train()` 调用后 EMA 模型应进入 train 模式，或修改测试定义为 EMA 始终 eval。建议按测试修复：`BaseTrainingStrategy._set_model_mode(training)` 对 `ema_model` 使用同样的 `training` 状态。
验收：`tests/test_strategy_base.py::StrategyBaseTest::test_base_strategy_can_enable_optional_ema_support` 通过。
5. 修复 Trainer 对精简 Namespace 的兼容
目标文件：`core/train.py`
修复：读取 `early_stopping` 和 `amp` 时使用默认值兜底，避免测试构造的轻量 Namespace 报 `AttributeError`。
验收：`tests/test_train_entrypoint.py` 当前失败项通过。
## P1：补齐 checkpoint 与 resume
### 现状
`--use_checkpoint` 已定义但训练入口未使用；checkpoint 未保存 optimizer、scheduler、scaler、epoch、RNG 状态；验证模式下 `model_final.pth` 是 0 字节 marker。
### 修复项
1. 实现 resume
目标文件：`core/train.py`
修复：当 `--use_checkpoint` 为真时，加载 `model_latest.pth` 或 `model_best.pth`，恢复 `iter_num`、`best_performance`、`optimizer`、`lr_scheduler`、`scaler`。
验收：中断训练后再次运行能从原 iter 继续，日志记录恢复点。
2. 保存完整训练状态
目标文件：`core/train.py`
修复：checkpoint 字段包含 `model_state`、`optimizer_state`、`scheduler_state`、`scaler_state`、`args`、`iter_num`、`best_iter`、`best_performance`、`rng_state`。
验收：恢复后学习率、AMP scaler、best 指标不重置。
3. 替换 0 字节 final marker
目标文件：`core/train.py`、`core/test.py`、`tests/test_train_entrypoint.py`
修复：`model_final.pth` 始终保存可加载模型；若需要标记验证模式，新增 `TRAIN_DONE` 文件。
验收：`python -m core.test --pth final` 能加载 final checkpoint。
## P2：默认行为稳定化
### 修复项
1. 增加关闭 AMP 和 compile 的 CLI
目标文件：`core/args.py`、`core/train.py`
修复：将 `--amp`、`--compile` 改为显式布尔解析，支持 `--amp false`、`--compile false`，或新增 `--no_amp`、`--no_compile`。
验收：CPU/旧 GPU 环境可关闭 AMP 和 compile。
2. 优化已有结果跳过逻辑
目标文件：`core/runtime.py`、`core/train.py`
修复：不要用 0 字节 `model_final.pth` 参与“训练完成”判断；使用可加载 checkpoint 或 `TRAIN_DONE`。
验收：损坏 checkpoint 不会被误判为完成。
3. 统一 README 与代码默认值
目标文件：`README.md`
修复：同步默认 `lr`、`pretrain`、`model`、`rgb`、`final/best` checkpoint 语义。
验收：README 快速命令无需额外补参数即可与默认路径一致。
## P3：回归测试补强
### 新增测试
1. `--use_checkpoint` 恢复 optimizer、scheduler、scaler、iter。
2. `model_final.pth` 可被 `core.test` 加载。
3. 显式 `--pretrain none` 与默认 `--pretrain resnet` 路径分离。
4. `--compile false` 和 `--amp false` 参数生效。
5. 损坏或空 checkpoint 不会触发跳过训练。
## 推荐执行顺序
1. 修 P0，确保 `python -m pytest -q` 全绿。
2. 修 P1，保证训练中断可恢复，final checkpoint 可测试。
3. 修 P2，降低跨机器运行风险。
4. 补 P3，锁住行为避免再次漂移。
## 最终验收命令
```bash
python -m pytest -q
python -m pytest tests/test_args_runtime.py tests/test_runtime_paths.py tests/test_train_entrypoint.py tests/test_strategy_base.py -q
```
