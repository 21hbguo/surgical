# Way & Exp 名称对照表

## 主实验（Task 1/2/3）

| way | exp | 说明 |
|-----|-----|------|
| fully | Fully | 全监督上界 |
| mt | MT | Mean Teacher |
| uamt | UAMT | Uncertainty-aware Mean Teacher |
| urpc | URPC | Uncertainty-Rectified Pyramid Consistency |
| cps | CPS | Cross Pseudo Supervision |
| mms | MMS | Min-Max Similarity |
| unimatch | UniMatch | UniMatch (reimpl.) |
| unimatch_official | UniMatch | UniMatch (official) |
| segmatch_official | SegMatch | SegMatch (official) |
| u2pl | U2PL | U2PL (official) |
| corrmatch | CorrMatch | CorrMatch (official) |
| cwbass | CW-BASS | CW-BASS (official) |
| georisk_spc | GeoRiskSPC | GeoRisk-SPC (UNet backbone) |
| georisk_spc_dgv4 | GeoRiskSPC_DGv4 | GeoRisk-SPC-DG (our method) |
| mt_depth_guider_v4 | MT_depth_guider_v4 | MT + DepthGuiderV4 |
| mt_depth_guider_v1_2 | MT_depth_guider_v1_2 | MT + DepthGuiderV1.2 |
| ternaus | TernausNet16 | TernausNet16 |

## 消融实验（Task 1, 20% labels）

| way | exp | 说明 |
|-----|-----|------|
| georisk_spc_dgv4 | Ablation_depth_uncertainty | Depth + Uncertainty only |
| georisk_spc_dgv4 | Ablation_no_risk | w/o risk localization |
| georisk_spc_dgv4 | Ablation_dg_no_risk | DG encoder w/o risk |
| georisk_spc_dgv4 | Ablation_no_cons | w/o L_cons |
| georisk_spc_dgv4 | Ablation_tau_r_0.7 | τ_r=0.7 |
| georisk_spc_dgv4 | Ablation_random_mask | Random mask (τ_r=1.0) |
| georisk_spc_dgv4 | Ablation_depth_only | Depth only |
| georisk_spc_dgv4 | Ablation_conflict_only | Conflict only |
| georisk_spc_dgv4 | Ablation_tau_c_0.7 | τ_c=0.7 |
| georisk_spc_dgv4 | Ablation_uncertainty_only | Uncertainty only |
| georisk_spc_dgv4 | Ablation_tau_r03_c07 | τ_r=0.3, τ_c=0.7 |
| georisk_spc_dgv4 | Ablation_tau_r_0.3 | τ_r=0.3 |
| georisk_spc_dgv4 | Ablation_tau_c_0.95 | τ_c=0.95 |
| georisk_spc_dgv4 | Ablation_no_bd | w/o L_bd |

## 额外实验

| way | exp | 说明 |
|-----|-----|------|
| mt | MT_s42/43/44 | MT 不同随机种子 |
| uamt | UAMT_s42/43/44 | UAMT 不同随机种子 |
| mt | SR_MT_s42/43/44 | Self-Region MT |
| mt_depth_guider_v4 | GAC_pred_only_s42 | GAC pred only |

## 命令格式

```bash
python -m core.train --task {1|2|3} --way {way} --exp "endovis2017/{exp}" --labeled_num {5|10|20|40} --fold {0|1|2|3}

# 需要 depth 的方法加：
--use_depth 1
```
