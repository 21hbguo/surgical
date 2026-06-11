# Global Rules
- 本项目是医学分割训练实验代码，第一目标是可读、可跑、可复现、可对比，不写通用训练框架，不做无实验收益的工程化包装
- 代码改动必须让训练主线一眼可见：参数解析、随机种子、数据划分、DataLoader、模型、优化器、loss、训练循环、验证、checkpoint、指标记录
- 不生成任何注释；不添加无关空格、换行、注释；代码紧凑，无空行分隔
- 变量命名格式保持前后一致，禁止无意义重命名，禁止把 `args.xxx` 赋值给只用一次的同义变量
- 禁止为只调用1次的代码块单独定义函数、类、dataclass、helper、wrapper、装饰器；只有被复用2次以上或能显著消除训练主线重复的逻辑才允许抽出
- 禁止嵌套函数、回调式封装、钩子式跳转、隐式注册链路；PyTorch 必需接口除外
- 参数只解析1次，派生参数只归一化1次；禁止在 train/test/runtime/strategy 之间重复解析、重复推断、重复改写同一参数
- `args` 字段含义必须唯一稳定，禁止同一字段在不同阶段表示不同含义；新增参数必须直接服务数据、模型、loss、优化、验证、保存或复现实验
- 训练入口保持线性流程：parse args -> finalize args -> seed -> folds -> dataloader -> model -> optimizer -> strategy/loss -> train -> validate -> save
- 策略文件只写该策略的核心差异：前向、监督损失、无监督损失、权重调度、EMA/伪标签/扰动逻辑；禁止把通用训练流程复制进策略
- loss 返回必须包含 `total` 和关键子项，日志字段名固定且可直接对比；禁止只记录总 loss 后隐藏核心实验信号
- 数据相关逻辑必须显式保留：task、fold、labeled/unlabeled 划分、sampling、normalize、depth 通道、depth_uint、resize、num_classes
- 每次训练必须保存本轮 args、labeled/unlabeled/val list、best/final checkpoint、metrics.csv；验证关闭时必须明确只保存 final
- checkpoint 规则固定：有验证保存 `model_best.pth` 和 `model_final.pth`，无验证只保存 `model_final.pth`
- 指标优先级固定：Dice 为主指标，HD95/IoU/ASD/NSD 按已有实现保留；新增指标必须写入 CSV 且不破坏原字段
- 随机性必须统一由 `seed` 控制，涉及 sampler、worker、numpy、random、torch、cuda 的地方不得各自写死不同 seed
- 新增模型或策略必须接入现有 `models.factory` 和 `strategies.specs`，不得新增第二套 registry、第二套参数系统或第二套训练入口
- 删除冗余优先于新增包装：发现重复解析、冗余赋值、死分支、未使用参数、只转发一层的函数，直接合并或删除
- 改训练代码后至少验证对应 parser、单步 loss、checkpoint/metrics 路径；无法运行时说明缺失环境或数据
- 答复简洁专业、紧凑准确；输出仅核心内容，不冗余表述；不使用“可能/大概/也许”等不确定表述，若不确定先查证再回答
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 78
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/guo/project/other/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'	' '$1=="repo_root"{print $2; exit}' "/home/guo/project/ssl4mis/code_all_vibe_v2/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/guo/project/other/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/home/guo/project/ssl4mis/code_all_vibe_v2" --reconcile`
For copied Codex installs, use:
`bash /home/guo/project/other/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/home/guo/project/ssl4mis/code_all_vibe_v2"`
<!-- ARIS-CODEX:END -->
