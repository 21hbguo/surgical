# ssl4mis/code_all 多维度审批报告
## 1. 审批结论
- 审批时间: 2026-05-13
- 审批范围: `core/` `strategies/` `models/` `data/` `utils/` `tests/` `scripts/`
- 审批方式: 静态代码审查 + 关键路径复现 + 单元测试回归
- 当前结论: 不通过
- 通过条件: 先完成全部 H 级问题修复并回归通过
## 2. 量化评分
| 维度 | 分数(10) | 结论 |
|---|---:|---|
| 架构分层与扩展性 | 7.0 | 通过 |
| 训练流程可靠性 | 5.0 | 风险 |
| 推理评测可靠性 | 3.0 | 不通过 |
| 数据管线稳健性 | 5.0 | 风险 |
| 指标计算正确性 | 4.0 | 风险 |
| 配置契约一致性 | 3.0 | 不通过 |
| 可复现性 | 5.0 | 风险 |
| 测试工程质量 | 4.0 | 风险 |
| 可维护性 | 5.0 | 风险 |
| 文档与实现一致性 | 4.0 | 风险 |
| 综合评分 | 4.5 | 不通过 |
## 3. 回归测试结果
- 执行命令: `pytest -q`
- 结果: `135 passed, 10 failed`
- 失败集中在参数默认值与路径命名契约漂移:
  - `tests/test_args_runtime.py`
  - `tests/test_runtime_paths.py`
  - `tests/test_task_dataset_selection.py`
## 4. H 级问题
### H-01 验证模式下 `model_final.pth` 被写成空文件，`final` 测试路径直接失效
- 证据:
  - `core/train.py:403-408` 对 `suffix=="final" and self.has_val` 直接写零字节文件
  - `core/test.py:563-565` `final` 路径固定加载 `model_final.pth`
  - `utils/common.py:230-233` `torch.load()` 直接读取 checkpoint
  - `tests/test_train_entrypoint.py:86-107` 显式断言该文件应为 0 字节
- 影响:
  - 开启验证训练后，`--checkpoint-type final` 必然无法作为可加载模型使用
  - 训练完成状态与可推理状态解耦，产物语义不一致
### H-02 推理主路径绕过策略 `validation_step`，导致多策略输入组装错误
- 证据:
  - `core/test.py:112-133` `_predict_logits` 在存在 `strategy.model` 时按通用逻辑拼接输入
  - `core/test.py:566,579-588` `run_one_fold` 固定传入 `strategy`，走上述通用分支
  - `strategies/semi_mt_depth_teacher_v1.py:136-139` 验证只喂 `image`，不拼 depth
  - `strategies/fully_rgb_masking_depth_v1.py:84-94` 验证需走互补融合输入，不是简单拼接
  - `strategies/semi_rdnet.py:237-243` 验证需按 `_match_expected_channels` 走 RGB 分支
- 复现:
  - 使用 `mt_depth_teacher_v1 + use_depth=3` 的最小 case 调用 `run_one_fold`，触发通道不匹配运行时错误
- 影响:
  - 推理结果对策略实现不忠实
  - 多策略在测试阶段存在直接崩溃风险
### H-03 RDNet 策略输入契约不闭合，默认配置可直接触发运行错误
- 证据:
  - `strategies/specs.py:96` `rdnet` 未约束 `in_chns`，沿用 metadata+depth 推导
  - `strategies/specs.py:153` `use_depth=3` 时 `depth_in_chns=3`
  - `strategies/base_strategy.py:30` `use_depth=3` 时仅取 `depth3`
  - `strategies/semi_rdnet.py:132,137` 同时强依赖 `depth1`
- 复现:
  - `RDNetStrategy.compute_loss` 在 `use_depth=3` 且未提供 `depth1` 时触发 `TypeError: 'NoneType' object is not subscriptable`
- 影响:
  - 策略配置面向用户不可自解释，错误在运行期才暴露
### H-04 默认参数与测试基线已发生系统性漂移，CI 处于失败状态
- 证据:
  - `core/args.py:117` 默认 `lr=3e-4`
  - `tests/test_args_runtime.py:81` 断言默认 `lr=3e-5`
  - `core/args.py:37` 默认模型随 `pretrain=none` 解析为 `unet*`
  - `tests/test_args_runtime.py:104,118,126,194,252` 断言默认/派生模型为 `resnet*`
  - `pytest -q` 当前 10 例失败
- 影响:
  - 主干行为定义不一致
  - 历史实验路径命名、汇总 CSV 统计键将持续漂移
## 5. M 级问题
### M-01 Precision/Recall 除零返回 1.0，空预测被计为满分
- 证据: `core/test.py:424-425,441-442`
- 影响: 完全未检出时 Precision/Recall 统计被系统性高估
### M-02 test fold 过滤对样本命名格式硬编码，非 `xxx_数字_xxx` 即崩溃
- 证据:
  - `data/dataset.py:52-55` 直接 `int(item.split("_")[1])`
  - 最小复现对 `frame0001` 触发 `IndexError`
- 影响: 新数据集命名稍有变化即无法测试
### M-03 `finalize_test_args` 中 `pth` 与 `checkpoint_type` 状态不一致
- 证据:
  - `core/args.py:67` `checkpoint_type` 在 `no_val` 下强制 `final`
  - `core/args.py:68` `args.pth` 仍保留 `requested`（默认 `best`）
- 影响: 参数日志和内部状态表达不一致，增加排障歧义
### M-04 标签目录定位缺少保护分支，错误信息不可诊断
- 证据: `utils/common.py:58-67` 直接 `return matches[0]`
- 影响: 缺失目录时抛 `IndexError`，定位成本高
### M-05 导出的 labeled/unlabeled list 含重复样本，不等价于真实集合
- 证据:
  - `core/train.py:143-144` 会按 batch size 扩增索引
  - `core/train.py:247-248` 原样写入列表文件
- 影响: 产物更像 sampler 输入而非数据划分基准，影响实验审计
### M-06 CSV 并发写锁为可选依赖，缺包时退化为无锁写入
- 证据: `core/testing/export.py:126-129`
- 影响: 多进程/多实验同时落盘时存在竞态覆盖风险
## 6. L 级问题
### L-01 `core/test_view.py` 与 `core/test.py` 重复且未被引用
- 证据:
  - 文件内容一致
  - 全仓检索无引用
- 影响: 后续维护易出现双文件漂移
### L-02 fold 序列映射仅支持 4 折硬编码
- 证据: `core/test.py:69-77`
- 影响: 新数据集折数扩展时功能退化为无映射
### L-03 验证集 DataLoader 未设置 `worker_init_fn`
- 证据:
  - 训练 loader: `core/train.py:165-171`
  - 验证 loader: `core/train.py:172`
- 影响: 与训练 seed 传播策略不对齐
### L-04 文档中的 `model_final.pth` 语义与实现不一致
- 证据:
  - 文档声明训练产物包含 `model_final.pth`: `README.md:130-137`
  - 实现中该文件在有验证场景为零字节标记: `core/train.py:405-408`
- 影响: 用户按文档使用 `final` 推理会踩坑
## 7. 维度结论
### 7.1 架构层面
- 优点: 策略注册与模型注册边界清晰，扩展路径明确
- 结论: 架构可扩展，但运行契约未闭环
### 7.2 训练与推理一致性
- 结论: 不一致点集中在测试输入装配与 checkpoint 语义，已构成主流程阻断
### 7.3 指标可信度
- 结论: Dice/IoU 主路径可用，Precision/Recall 存在系统性高估问题
### 7.4 可复现性与工程稳定性
- 结论: 当前测试基线失败，默认参数定义需先统一再谈可复现
## 8. 修复优先级与验收标准
### P0（本周内）
- 修复 H-01/H-02/H-03/H-04
- 验收:
  - `pytest -q` 全绿
  - `--checkpoint-type final` 可正常加载并完成一次完整推理
  - `mt_depth_teacher_v1` `fully_rgb_masking_depth_v1` `rdnet` 三策略各跑通 1 fold 测试
### P1（下周）
- 修复 M-01/M-02/M-03/M-04
- 验收:
  - 指标除零策略与定义文档一致
  - 非下划线样本命名可通过 fold 过滤流程
  - 参数日志与运行状态完全一致
### P2（后续）
- 修复 M-05/M-06 与全部 L 级问题
- 验收:
  - 数据划分导出文件可直接用于审计
  - 并发导出无覆盖
  - 文档与实现一致
## 9. 最终审批意见
- 当前版本不具备“稳定可复现实验主干”的放行条件
- 建议执行 P0 修复后再发起复审
