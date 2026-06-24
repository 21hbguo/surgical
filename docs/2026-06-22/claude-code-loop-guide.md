# Claude Code `/loop` 完整指南

> 基于官方文档及社区资料整理，2026-06-22

## 概述

`/loop` 是 Claude Code 内置的 **bundled skill**，用于在当前会话中按时间间隔重复运行提示词。它本质上是一个会话级的调度原语，补充了 hooks、skills 和 sub agents 的能力。

- **别名**: `/proactive`
- **核心思想**: 告诉 Claude "每隔 N 分钟做某件事"，然后你就可以去做别的事

## 语法

```
/loop [interval] [prompt]
```

### 三种调用方式

| 提供的内容 | 示例 | 行为 |
|-----------|------|------|
| 间隔 + 提示 | `/loop 5m check the deploy` | 按固定间隔运行 |
| 仅提示 | `/loop check the deploy` | Claude 动态选择间隔（1min ~ 1h） |
| 仅间隔或无参数 | `/loop` | 运行内置维护提示或 `.claude/loop.md` |

### 时间单位

| 单位 | 符号 | 说明 |
|------|------|------|
| 秒 | `s` | 向上取整到最近的分钟 |
| 分钟 | `m` | 直接映射 |
| 小时 | `h` | 直接映射 |
| 天 | `d` | 直接映射 |

默认间隔为 **10 分钟**。

### 示例

```bash
# 固定间隔
/loop 5m check if the deployment finished
/loop 30m scan error logs for new FATAL entries
/loop 1d summarize all commits from the last 24 hours

# 动态间隔（Claude 自行决定节奏）
/loop check whether CI passed and address any review comments

# 嵌套 skill 调用
/loop 20m /review-pr 1234
/loop 1h /security-review

# 一次性提醒（自然语言）
remind me at 3pm to push the release branch
in 45 minutes, check whether the integration tests passed
```

## 自定义默认提示: loop.md

当不提供提示词时，`/loop` 会读取 `loop.md` 文件：

| 路径 | 作用域 |
|------|--------|
| `.claude/loop.md` | 项目级，优先级更高 |
| `~/.claude/loop.md` | 用户级，项目未定义时使用 |

示例 `.claude/loop.md`:
```
Check the `release/next` PR. If CI is red, pull the failing job log,
diagnose, and push a minimal fix. If new review comments have arrived,
address each one and resolve the thread. If everything is green and
quiet, say so in one line.
```

## 底层机制

`/loop` 底层使用三个原生工具：

| 工具 | 功能 |
|------|------|
| `CronCreate` | 创建定时任务，接受 5 字段 cron 表达式 |
| `CronList` | 列出当前会话的所有定时任务 |
| `CronDelete` | 按 ID 取消任务 |

可以自然语言调用：
```
what scheduled tasks do I have?
cancel the deploy check job
```

### 调度行为

- **首次立即执行**：创建后马上运行第一次
- **仅空闲时触发**：Claude 忙碌时等待，空闲后触发一次（不补发错过的次数）
- **上下文累积**：每次迭代在同一对话中运行，Claude 保留之前所有迭代的上下文
- **按 Esc 停止**：在等待中的循环上按 Esc 可终止

### Jitter（抖动）

为避免所有会话同时请求 API，调度器会添加随机偏移：
- 周期任务：最多延迟周期的 10%，上限 15 分钟
- 一次性任务：整点/半点触发的任务最多提前 90 秒
- **建议**：精确时间场景避免 `:00` 和 `:30`，用 `3 9 * * *` 代替 `0 9 * * *`

### Cron 表达式参考

```
minute hour day-of-month month day-of-week

*/5 * * * *    → 每 5 分钟
0 * * * *      → 每小时整点
7 * * * *      → 每小时第 7 分钟
0 9 * * *      → 每天 9:00
0 9 * * 1-5    → 工作日 9:00
30 14 15 3 *   → 3月15日 14:30
```

## 约束条件

| 约束 | 详情 |
|------|------|
| **会话作用域** | 关闭终端/退出会话即丢失，无恢复机制 |
| **自动过期** | 循环任务 **7 天**后自动过期（部分文档提到 3 天，以官方文档为准） |
| **任务上限** | 每个会话最多 **50 个**定时任务 |
| **分钟粒度** | 基于 cron，低于 1 分钟的间隔向上取整 |
| **无补发** | 忙碌时错过的间隔只补发一次，不是每次补发 |
| **禁用开关** | `CLAUDE_CODE_DISABLE_CRON=1` 可完全禁用调度器 |

## 典型使用场景

### 1. 部署监控
```bash
/loop 5m check if the staging deployment at port 3000 is responding, and tell me the HTTP status code
```

### 2. CI 流水线监控
```bash
/loop 3m check if the CI run on the current branch passed. If it did, stop looping and summarize the results.
```

### 3. PR 审查轮询
```bash
/loop 10m /review-pr 1234
```

### 4. 错误日志扫描
```bash
/loop 2h scan the error logs in ./logs/app.log for new FATAL entries since the last check. If any are fixable, open a PR with the fix.
```

### 5. 每日团队摘要
```bash
/loop 1d summarize all commits from the last 24 hours across the main branch, group by author
```

### 6. 值班工单分类
```bash
/loop 15m check for new tickets in the on-call queue, summarize each, propose a solution
```

### 7. 定时提醒
```bash
remind me at 4:45 PM to tag the release candidate before the deploy window closes
```

## /loop vs 其他自动化方案

| 需求 | 最佳方案 |
|------|----------|
| 工作时每隔 N 分钟做某事 | `/loop` |
| 响应工具调用或生命周期事件 | Hooks |
| 按需运行可复用提示 | Skills |
| 委托一次性任务给专家 | Sub Agents |
| CI 中可靠地按计划运行 | GitHub Actions |
| 协调多个 agent | Agent Teams |

### 调度方案对比

| | Cloud (`/schedule`) | Desktop | `/loop` |
|---|---|---|---|
| 运行位置 | Anthropic 云 | 你的机器 | 你的机器 |
| 需要开机 | 否 | 是 | 是 |
| 需要打开会话 | 否 | 否 | 是 |
| 跨重启持久化 | 是 | 是 | `--resume` 可恢复（未过期） |
| 访问本地文件 | 否（全新 clone） | 是 | 是 |
| 最小间隔 | 1 小时 | 1 分钟 | 1 分钟 |

## 最佳实践

### 上下文管理
- 每次迭代会累积上下文，长时间运行会触发 compaction 或达到上下文限制
- 保持每次迭代输出精简
- 定期使用 `/compact` 压缩上下文

### 与 Sub Agent 组合
用 sub agent 做重活，主上下文保持干净：
```bash
/loop 10m /post-deploy-monitor
```
其中 `/post-deploy-monitor` 是配置了 `model: sonnet` 的 skill，返回单行摘要，sub agent 的完整上下文在返回后丢弃。

### 安全提醒
- **Token 成本累积**：每 10 分钟运行一次、持续 4 小时 = 24 次迭代
- **写操作危险**：会修改文件、提交代码的 loop 被遗忘后会无人值守地持续执行
- **定期检查** `/tasks`：查看活跃的循环任务，及时停止不需要的

### 不适合的场景
- 需要任务在重启后存活 → 用 GitHub Actions 或 Desktop scheduled tasks
- 团队关键自动化 → 用 GitHub Actions
- 需要精确时间保证 → 用系统 cron + `claude -p`
- 事件驱动任务 → 用 Hooks

## 跨会话自主循环（高级模式）

通过 stop hook 可实现跨会话的自主循环：

1. **任务模板**：定义目标和限制
2. **Stop hook**：会话结束时检查任务文件是否存在，自动重启
3. **Kill switch**：`touch ~/.claude/autonomous/STOP` 立即终止

这是一种社区模式，适合需要长时间自主运行的场景（如自动修复测试失败）。

## 环境变量

| 变量 | 作用 |
|------|------|
| `CLAUDE_CODE_DISABLE_CRON=1` | 完全禁用调度器 |

## 参考来源

- [官方文档: Run prompts on a schedule](https://code.claude.com/docs/en/scheduled-tasks)
- [官方文档: Commands](https://code.claude.com/docs/en/commands)
- [Claude Code Guide: Loop](https://claude-code-guide.org/loop/)
- [Verdent: How to Use Claude Code /loop](https://www.verdent.ai/guides/claude-code-loop-command)
- [Developers Digest: The Definitive Guide to Loop Engineering](https://www.developersdigest.tech/blog/loop-engineering-definitive-guide)
