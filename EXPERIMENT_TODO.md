# 实验待办清单

## 优先级说明
- P0: 必须完成（论文表格需要）
- P1: 重要（增强论文说服力）
- P2: 可选（进一步验证）

---

## 一、基线对比实验（P0 - 进行中）

### 已完成 ✅
| 方法 | 5% | 10% | 20% | 40% | 状态 |
|------|-----|-----|-----|-----|------|
| CPS | 85.39 | 81.38 | 89.88 | 87.92 | ✅ |
| MMS | 85.75 | 87.44 | 87.66 | 84.35 | ✅ |
| UniMatch_official | 85.29 | 86.16 | 86.61 | 88.87 | ✅ |
| SegMatch_official | 86.42 | ⏳ | ⏳ | ⏳ | 进行中 |

### 队列进度
- SegMatch_official 10% f0 训练中
- 剩余: 10% f1-3, 20% × 4, 40% × 4

### 新增 2024-2026 基线（P0 - 待运行）

基于原始 GitHub 代码忠实复现，统一使用 UNet backbone + Adam(lr=1e-4) + 30000 iter。

| 方法 | 年份 | 会议 | 核心思想 | 状态 |
|------|------|------|----------|------|
| U2PL | 2022 | CVPR | 不可靠伪标签对比学习 + 内存银行 | 代码完成 |
| CorrMatch | 2024 | CVPR | 相关性匹配标签传播 | 代码完成 |
| CW-BASS | 2025 | IJCNN | 置信度加权边界感知学习 | 代码完成 |

运行命令（每方法 4 标签率 × 4 folds）：
```bash
# U2PL
python -m core.train --task 1 --way u2pl --exp endovis2017/U2PL --labeled_num {5,10,20,40} --fold {0,1,2,3}

# CorrMatch
python -m core.train --task 1 --way corrmatch --exp endovis2017/CorrMatch --labeled_num {5,10,20,40} --fold {0,1,2,3}

# CW-BASS
python -m core.train --task 1 --way cwbass --exp endovis2017/CWBASS --labeled_num {5,10,20,40} --fold {0,1,2,3}
```

---

## 二、消融实验（P0 - 待运行）

在 20% 标签率下进行消融实验，使用 GeoRisk-SPC-DG 配置。

### 2.1 组件消融
| 实验 | 命令参数 | 说明 |
|------|----------|------|
| w/o L_cons | `--risk_cons_weight 0` | 移除高风险一致性损失 |
| w/o L_bd | `--risk_bd_weight 0` | 移除边界一致性损失 |
| Random mask | 需要代码修改 | 用随机mask替代风险图 |
| w/o risk localization | 需要代码修改 | 移除整个风险定位机制 |

### 2.2 风险源分解
| 实验 | 命令参数 | 说明 |
|------|----------|------|
| Depth only | 需要代码修改 | 仅使用深度不连续性 |
| Uncertainty only | 需要代码修改 | 仅使用教师不确定性 |
| Conflict only | 需要代码修改 | 仅使用几何-语义冲突 |
| Depth + Uncertainty | 需要代码修改 | 移除冲突项 |

### 2.3 架构消融
| 实验 | 命令参数 | 说明 |
|------|----------|------|
| MT + RGB-D concat | `--way mt --use_depth 1` | Mean Teacher + 深度拼接 |
| MT + DG encoder | `--way mt_depth_guider_v4` | Mean Teacher + DG编码器 |
| GeoRisk-SPC w/o DG | `--way georisk_spc --use_depth 1` | 已完成: 88.72% |
| DG w/o risk | 需要代码修改 | DG编码器但无风险监督 |

### 2.4 阈值敏感性
| 实验 | 命令参数 | 说明 |
|------|----------|------|
| tau_r = 0.3 | `--risk_tau_r 0.3` | 更严格的高风险判定 |
| tau_r = 0.7 | `--risk_tau_r 0.7` | 更宽松的高风险判定 |
| tau_c = 0.8 | `--risk_tau_c 0.8` | 更低的置信度阈值 |
| tau_c = 0.95 | `--risk_tau_c 0.95` | 更高的置信度阈值 |

---

## 三、实验执行计划

### Phase 1: 完成基线（当前）
等待 SegMatch_official 队列完成（预计还需 ~8 小时）

### Phase 2: 组件消融（P0）
在 20% 标签率下运行 4 个实验：
```bash
# w/o L_cons
python -m core.train --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_ablation --labeled_num 20 --fold -1 --use_depth 1 --risk_cons_weight 0

# w/o L_bd
python -m core.train --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_ablation --labeled_num 20 --fold -1 --use_depth 1 --risk_bd_weight 0

# tau_r = 0.3
python -m core.train --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_ablation --labeled_num 20 --fold -1 --use_depth 1 --risk_tau_r 0.3

# tau_r = 0.7
python -m core.train --task 1 --way georisk_spc_dgv4 --exp endovis2017/GeoRiskSPC_ablation --labeled_num 20 --fold -1 --use_depth 1 --risk_tau_r 0.7
```

### Phase 3: 阈值敏感性（P1）
运行 4 个阈值实验

### Phase 4: 架构消融（P1）
运行 MT + RGB-D 和 MT + DG 实验

---

## 四、论文更新计划

### 需要更新的内容
1. **Table 1**: 填入 CPS 和 MMS 结果（替换 "--"）
2. **Table 2**: 填入所有消融实验结果
3. **分析文字**: 根据消融结果更新分析段落

### 预期结果趋势
- w/o L_cons: Dice 应下降 1-2%，验证一致性损失的贡献
- w/o L_bd: Dice 应下降 0.5-1%，验证边界损失的贡献
- Random mask: Dice 应下降 2-3%，验证风险引导的重要性
- w/o risk: Dice 应下降 3-5%，验证整个风险机制的贡献

---

## 五、时间估算

| 阶段 | 实验数 | 单实验时间 | 总时间 |
|------|--------|------------|--------|
| 基线完成 | 12 folds | ~30min/fold | ~6h |
| 组件消融 | 4 | ~3h/实验 | ~12h |
| 阈值敏感性 | 4 | ~3h/实验 | ~12h |
| 架构消融 | 2 | ~3h/实验 | ~6h |
| **总计** | | | **~36h** |

---

## 六、注意事项

1. **GPU资源**: 基线队列完成后，GPU资源释放，可并行运行消融实验
2. **checkpoint管理**: 消融实验使用不同的exp名称，避免覆盖已有结果
3. **结果收集**: 所有实验完成后，统一更新论文表格
4. **代码修改**: Random mask 和风险源分解需要修改 `semi_georisk_spc.py`
