# Session Backup - 2026-05-31

## 当前状态

### GPU 问题
- NVIDIA 驱动被 unattended-upgrades 自动升级: 580.126.09 → 580.159.03
- 内核模块版本不匹配，CUDA 不可用
- 需要 `sudo reboot` 重启恢复

### GeoRisk-SPC 实验（全部训练完成，待测试）
| 配置 | 5% | 10% | 20% | 40% | 测试 |
|------|-----|-----|-----|-----|------|
| GeoRiskSPC plain | ✅ | ✅ | ✅ | ✅ | ❌ 未测试 |
| GeoRiskSPC DGv4 | ✅ | ✅ | ✅ | ✅ | ❌ 未测试 |

测试命令：
```bash
for pct in 5 10 20 40; do
    python -m core.test --task 1 --way georisk_spc --exp endovis2017/GeoRiskSPC --labeled_num $pct --fold -1 --use_depth 1
    python -m core.test --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_DGv4 --labeled_num $pct --fold -1 --use_depth 1
done
```

### 基线训练队列
| 方法 | 5% | 10% | 20% | 40% |
|------|-----|-----|-----|-----|
| CPS | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| MMS | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| UniMatch_official | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| SegMatch_official | ✅ 4/4 | ✅ 4/4 | ⏳ 1/4 | ❌ 0/4 |

恢复命令: `bash run_missing_baselines.sh`

### 新基线（代码已就绪，待运行）
- U2PL: `strategies/semi_u2pl.py` ✅ 已通过 Codex 审查并修复
- CorrMatch: `strategies/semi_corrmatch.py` ✅ 已通过 Codex 审查并修复
- CW-BASS: `strategies/semi_cwbass.py` ✅ 已通过 Codex 审查并修复

运行命令：
```bash
for way in u2pl corrmatch cwbass; do
    for pct in 5 10 20 40; do
        for fold in 0 1 2 3; do
            python -m core.train --task 1 --way $way --exp endovis2017/${way^^} --labeled_num $pct --fold $fold
        done
    done
done
```

### 本次 session 完成的工作
1. 发送 U2PL/CorrMatch/CW-BASS 代码给 Codex 审查
2. 修复了 5 个 P0 关键问题（详见会话摘要）
3. 验证三个策略 import 正常
4. 诊断 GPU 驱动版本不匹配问题
5. 停止了浪费 CPU 的训练进程

### 结果目录
- 路径: `/home/guo/project/ssl4mis/result_train/endovis2017_255_Samplinginterval/task1/`
- 结构: `{方法}/{pct}_labeled_lr1e-4_s_unet*/f{0-3}/model_best.pth`

### 论文相关
- `paper/results_table.md` — 实验结果汇总
- `paper/main.tex` — 论文 LaTeX 源码
- `EXPERIMENT_TODO.md` — 实验待办清单（含消融实验）
