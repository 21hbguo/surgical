# GeoRisk-SPC Baseline and Experiment Guidance for MICCAI/TMI 2026
## Executive Recommendation
GeoRisk-SPC is publishable only if the experimental story is tightened around three points: official recent baselines, depth-fair controls, and structural-error evidence. The current draft already has a coherent method, but the present baseline table is not yet sufficient for MICCAI 2026 or TMI 2026 because UniMatch and SegMatch are marked as simplified reproductions, CPS is listed in the project context but absent from the main table, and the method uses depth while most baselines are RGB-only.
The minimum defensible main comparison should include:
1. Sup-only
2. MT
3. UAMT or URPC
4. CPS
5. BCP or MC-Net+
6. UniMatch official
7. MMS
8. SegMatch official
9. GeoRisk-SPC
10. GeoRisk-SPC+DGv4
For TMI, add at least one additional surgical dataset beyond EndoVis 2017. EndoVis 2017 alone is weak for a journal claim in 2026. SegMatch evaluated Robust-MIS 2019, EndoVis 2017, and CholecInstanceSeg; reviewers will notice if GeoRisk-SPC stays only on EndoVis 2017.
## Context From Current Draft
The current `paper/main.tex` describes GeoRisk-SPC as a Mean-Teacher-style framework with geometry-aware risk localization from depth discontinuity, teacher uncertainty, and geometry-semantic conflict. Low-risk regions use hard pseudo-label supervision; high-risk regions use structural perturbation consistency and boundary consistency.
Current reported setup:
1. Dataset: EndoVis 2017 Task 1 binary instrument segmentation
2. Evaluation: 4-fold cross-validation
3. Label ratios: 5%, 10%, 20%, 40%
4. Backbone: UNet with 16 initial filters
5. Optimizer: Adam, LR 1e-4, weight decay 1e-4
6. Training: 30,000 iterations, AMP, early stopping
7. Current main table: Sup-only, MT, UAMT, URPC, simplified UniMatch, simplified SegMatch, GeoRisk-SPC, GeoRisk-SPC+DGv4
Main risk:
The table currently says GeoRisk-SPC is competitive with recent strong baselines, but those baselines are not official reproductions. A reviewer can reject the comparison even if GeoRisk-SPC is genuinely strong.
## Baseline Completeness for 2026
### P0 Baselines
These should be in the main paper or the core appendix.
| Baseline | Priority | Why reviewers expect it | Recommended handling |
|---|---:|---|---|
| Sup-only | P0 | Needed to quantify the value of unlabeled data. | Same labeled splits, same backbone, same augmentation budget. |
| MT | P0 | Canonical teacher-student baseline. | Keep. |
| UAMT | P0 | Medical SSL classic uncertainty baseline. | Keep or replace with URPC if space is tight, but keeping both is safer. |
| URPC | P0 | Medical SSL classic, widely used in SSL4MIS-style comparisons. | Keep. |
| CPS | P0 | Canonical cross-pseudo-supervision semantic segmentation baseline. | Add to main table; it is currently missing from `main.tex`. |
| UniMatch official | P0 | Strong weak-to-strong semantic segmentation baseline. | Use official logic: weak view, two strong views, CutMix, feature perturbation branch. |
| MMS | P0 | Surgical-tool-specific semi-supervised contrastive baseline. | Add. This is more relevant than many generic medical SSL methods. |
| SegMatch official | P0 | Surgical-instrument-specific 2025 baseline with adversarial strong augmentation. | Add official reproduction; do not rely on the current simplified version. |
| GeoRisk-SPC depth controls | P0 | Your method uses depth; reviewers need attribution. | Add MT+D, UniMatch+D, SegMatch+D or equivalent depth-fair controls. |
### P1 Strongly Recommended Baselines
These are not all mandatory, but at least one should be added if compute permits.
| Baseline | Why it matters | Include when |
|---|---|---|
| BCP | A strong CVPR 2023 semi-supervised medical image segmentation baseline built on Mean Teacher and bidirectional copy-paste. | Include one recent medical SSL baseline beyond URPC/UAMT. |
| MC-Net+ | Medical Image Analysis 2022; mutual consistency with multiple decoders and uncertainty-aware hard-region learning. | Include if you want a direct competitor to your dual-decoder/high-risk consistency design. |
| CrossMatch | Recent medical SSL method using image-level and feature-level perturbation plus knowledge distillation. | Include if reviewers challenge the novelty of structural perturbation consistency. |
| DiffRect | MICCAI 2024 latent diffusion label rectification for pseudo-label bias. | Include if you claim pseudo-label rectification; otherwise cite and keep as related work. |
| Surgivisor | 2024 transformer-based semi-supervised endoscopic instrument segmentation on EndoVis17/18. | Include as a surgical-specific reference; run only if you accept architecture mismatch. |
| UTMT | 2026 uncertainty-guided twin Mean Teacher for surgical endoscopic segmentation. | Cite in related work; run only if you add SAR-RARP50/CaDIS or want a recent twin-teacher surgical SSL comparator. |
| CorrMatch | CVPR 2024 general SSS label propagation via correlation matching. | Good appendix baseline, not surgical-specific. |
| UniMatch V2 | TPAMI 2025; DINOv2/ViT-based update of UniMatch. | Cite as modern SSS direction; only compare if you add a foundation-backbone experiment. |
### P2 Citation-Only or Appendix Baselines
Use these in related work unless the paper becomes a broad benchmark paper:
1. DTC, SASSNet, SS-Net, MCF, RCPS, FRCNet, MOST, M3HL
2. SemiVL and other vision-language/foundation SSS methods
3. SAM/SAM2/Surgical-DeSAM variants, unless you add a foundation-model comparison section
4. Weakly supervised or unsupervised surgical segmentation methods, unless the supervision setting is explicitly comparable
## What Is Missing Right Now
### Missing From The Current Main Table
1. CPS, despite being listed in the project baseline plan.
2. MMS, which is surgical-tool-specific and TMI-published.
3. SegMatch official, which is the most obvious surgical SSL baseline in 2025.
4. UniMatch official, because the current simplified reproduction is not enough.
5. A recent general medical SSL method such as BCP or MC-Net+.
6. Depth-fair baselines that separate depth input advantage from the GeoRisk risk-localization idea.
7. Boundary metrics and risk-localization metrics that directly support the claimed geometry contribution.
8. External dataset validation.
### Should You Include DiffRect?
DiffRect is not a must-have surgical baseline. It is a MICCAI 2024 semi-supervised medical image segmentation method for latent diffusion label rectification, evaluated on ACDC, MS-CMRSEG 2019, and Decathlon Prostate. It is relevant because it attacks pseudo-label bias and confirmation bias, but it is not surgical-specific and not naturally aligned with EndoVis instrument segmentation.
Decision:
Use DiffRect as related work, and include it as an appendix baseline only if implementation cost is moderate. It should not replace MMS or SegMatch.
### Should You Include CrossMatch?
CrossMatch is more relevant than DiffRect to GeoRisk-SPC because it combines image-level and feature-level perturbation strategies with knowledge distillation. It is a useful comparator for the structural perturbation consistency part of GeoRisk-SPC.
Decision:
Include CrossMatch if you have compute after P0 baselines and BCP/MC-Net+. If only one P1 perturbation baseline can be run, CrossMatch is more directly relevant than DiffRect.
### Should You Include Surgivisor?
Surgivisor is surgical-specific and evaluated on EndoVis17/18, so it belongs in related work and possibly in a secondary comparison. The issue is fairness: it is transformer-based and not a UNet-controlled baseline. If you include it, label the comparison as architecture-level or published-protocol comparison, not controlled UNet comparison.
Decision:
Cite Surgivisor in the main related work. Run it only if you add an architecture-mismatch table or a stronger-backbone section.
### Should You Include UTMT?
UTMT is a 2026 surgical endoscopic semi-supervised method based on twin Mean Teacher structures and uncertainty-weighted consistency. It is recent and thematically close to GeoRisk-SPC because both address unreliable pseudo-labels, but UTMT is evaluated on SAR-RARP50 and CaDIS rather than EndoVis 2017.
Decision:
Cite UTMT in related work. Treat it as a P1 baseline if you expand beyond EndoVis 2017 or if reviewers ask for 2026 surgical SSL comparators.
## Depth Fairness Controls
This is the most important experimental issue for GeoRisk-SPC.
The current paper states that GeoRisk-SPC uses depth in two ways: concatenated RGB-D input and risk-map construction. Baselines use RGB-only. This makes the main comparison vulnerable because improvements can be attributed to extra modality rather than the proposed risk localization and SPC.
Add at least these controls:
| Control | Purpose |
|---|---|
| Sup-only RGB-D | Measures the raw value of depth as input. |
| MT RGB-D | Checks whether Mean Teacher also benefits from the same extra channel. |
| UniMatch official RGB-D or SegMatch official RGB-D | Tests whether modern weak-to-strong baselines close the gap when given the same depth input. |
| GeoRisk-SPC with RGB input and depth used only for risk | Separates risk localization from depth-as-feature. |
| GeoRisk-SPC with RGB gradient/Canny risk instead of depth discontinuity | Shows the gain is not just edge detection. |
| GeoRisk-SPC with shuffled/noisy/blurred depth | Shows the method depends on meaningful geometry rather than arbitrary regularization. |
Best presentation:
1. Main table: RGB-only controlled baselines plus GeoRisk-SPC variants.
2. Depth control table: top baselines with RGB-D versus GeoRisk-SPC.
3. Ablation table: depth as input versus depth as risk prior.
## Experiment Standards
### Label Ratios
The 5%, 10%, 20%, 40% schedule is good for a label-efficiency curve, but it does not fully align with SegMatch, which reports 1:9 and 3:7 labeled-to-unlabeled ratios. Keep 5/10/20/40, and add 30% for direct SegMatch protocol alignment if compute allows.
Recommended ratios:
1. 5%: extreme low-label setting
2. 10%: aligns with 1:9
3. 20%: practical mid-label setting
4. 30%: aligns with 3:7 and helps SegMatch comparison
5. 40%: current high-label setting
6. 100% supervised: upper bound
### Cross-Validation And Splits
Use sequence-level splits only. Do not let nearby frames from the same sequence leak across train/validation/test. Save split files and report them.
The EndoVis 2017 protocol must be described precisely. The original challenge has 8 training videos with 225 annotated frames each, plus held-out test videos. If you use only the released 8 training sequences for 4-fold cross-validation, state that explicitly. Do not mix this with a 1350/450 split unless that exact protocol is documented and reproducible.
### Seeds
Four folds alone are not enough for modern claims when label subset selection is random. Use at least 3 seeds for label selection per ratio for the main methods.
Recommended minimum:
1. All P0 baselines: 4 folds x 1 fixed seed for all ratios
2. Top 4 methods: 4 folds x 3 seeds for 10%, 20%, and 30% or 40%
3. Ablations: 4 folds x 1 seed at 10% and 20%, or 3 seeds on one primary fold if compute is tight
### Training Recipe
Using UNet, Adam, LR 1e-4, 30,000 iterations, AMP, and early stopping is acceptable for a controlled comparison, but it is not enough as the only comparison.
Use two comparison styles:
1. Controlled recipe table: same UNet, same optimizer, same training length, same augmentations where applicable.
2. Strong reproduction table: official or carefully tuned settings for UniMatch, SegMatch, MMS, and GeoRisk-SPC.
This avoids the reviewer objection that strong baselines were weakened by forcing them into your recipe.
Early stopping must not use the test fold. If there is no clean validation sequence, prefer fixed-iteration training plus model averaging or select checkpoints using a validation subset created only from the training fold.
### Metrics
Report more than Dice. GeoRisk-SPC is a geometry/boundary method, so boundary metrics are required.
Minimum metrics:
1. Dice
2. IoU/Jaccard
3. Precision and recall
4. Boundary F1 or boundary IoU
5. NSD or HD95/ASSD
6. FPS, parameters, memory, and training overhead
Risk-localization metrics:
1. AUROC/AUPRC of risk map for pseudo-label error detection
2. Low-risk pseudo-label precision
3. High-risk coverage of pseudo-label errors
4. Risk-error correlation
5. Boundary-zone Dice, e.g. Dice within a fixed band around the ground-truth boundary
6. Error-type reduction: boundary drift, fragmentation, false merge, false positive specular region
### Statistical Reporting
Reviewers expect uncertainty reporting and significance testing, but p-values must respect video correlation.
Report:
1. Mean ± standard deviation across folds/seeds
2. 95% confidence intervals using bootstrap over sequences or fold-seed units
3. Paired test against the strongest baseline at the primary label ratios
4. Holm-Bonferroni correction if testing multiple baselines/ratios
5. Effect size, such as paired mean difference with CI
Avoid:
1. Frame-level p-values treating adjacent video frames as independent
2. Reporting only sample-level standard deviation without fold/seed variance
3. Claiming stability from a standard deviation before confirming whether it is image-level, fold-level, or seed-level
Recommended primary statistical claim:
At 10% and 20% labeled data, compare GeoRisk-SPC+DGv4 against SegMatch official and UniMatch official using paired sequence-level Dice and boundary metric differences, with 95% CI and Holm-corrected p-values.
## Required Ablations
### Risk Map Components
Run these at 10% and 20%:
1. uncertainty only
2. depth discontinuity only
3. geometry-semantic conflict only
4. uncertainty + depth
5. uncertainty + conflict
6. full risk map
7. random mask with same area
8. boundary dilation mask from RGB or pseudo-label boundary
9. Canny/RGB-gradient risk instead of depth discontinuity
### Supervision Strategy
Run:
1. all unlabeled pixels use hard pseudo-labels
2. low-risk hard pseudo-labels only, high-risk ignored
3. high-risk consistency only
4. no boundary consistency
5. global feature perturbation
6. risk-localized feature perturbation
7. no stop-gradient in clean branch
8. no ramp-up
### Depth Robustness
Run:
1. DPT/MiDaS depth
2. Depth Anything V2 depth if available
3. blurred depth
4. shuffled depth between images
5. depth normalized globally versus local relative normalization
6. varying Sobel/window size
This directly addresses the limitation that GeoRisk-SPC depends on depth quality.
### Threshold Sensitivity
The draft already notes threshold sensitivity. Turn it into a figure:
1. tau_r grid
2. tau_c grid
3. lambda_c grid
4. risk mask area versus Dice
5. adaptive threshold variant, e.g. top-k risk percentage per image
## Dataset Strategy
### MICCAI 2026
Minimum:
1. EndoVis 2017 with sequence-level 4-fold CV
2. One external surgical dataset or one held-out official test protocol
3. Strong qualitative and risk-localization evidence
Recommended:
1. EndoVis 2017
2. EndoVis 2018 collapsed to binary instrument segmentation
3. Robust-MIS 2019 if accessible
### TMI 2026
Minimum:
1. Two or three datasets
2. Cross-dataset generalization or robustness evaluation
3. Multiple seeds
4. Statistical tests and CIs
5. Runtime and clinical failure-mode analysis
Recommended:
1. EndoVis 2017
2. EndoVis 2018
3. Robust-MIS 2019 or CholecInstanceSeg
4. A corruption/robustness split for smoke, blur, specular highlight, low illumination, and occlusion
## Positioning Against UniMatch And SegMatch
Do not position GeoRisk-SPC as "stronger augmentation." UniMatch and SegMatch already own that framing.
Use this positioning:
GeoRisk-SPC addresses where pseudo-label supervision is trustworthy, while UniMatch and SegMatch mainly address how strongly to perturb unlabeled views. UniMatch applies weak-to-strong consistency with dual strong views and feature perturbation; SegMatch adapts FixMatch to surgical segmentation with adversarial strong augmentation. Both rely primarily on confidence and consistency under augmentation. GeoRisk-SPC adds an orthogonal surgical-scene prior: pseudo-label errors are spatially structured around instrument-tissue boundaries, occlusion interfaces, thin shafts, specular highlights, and depth discontinuities. Therefore, GeoRisk-SPC localizes risk before deciding whether to apply hard pseudo-label supervision or soft structural consistency.
Short version:
GeoRisk-SPC is a geometry-conditioned pseudo-label reliability framework, not another dual-strong-augmentation framework.
Main claim:
Existing weak-to-strong SSL methods perturb the image globally and filter pseudo-labels by confidence. GeoRisk-SPC estimates spatial pseudo-label risk using geometry and uncertainty, then applies region-adaptive supervision: hard labels in reliable regions and structural perturbation consistency in high-risk regions.
Best comparison language:
1. UniMatch improves the perturbation policy.
2. SegMatch improves surgical strong augmentation through adversarial perturbations.
3. GeoRisk-SPC improves the target reliability policy by deciding which regions deserve hard supervision and which require consistency-only treatment.
4. The methods are complementary; GeoRisk risk masks can be inserted into UniMatch/SegMatch-style training.
Avoid:
1. "We outperform UniMatch/SegMatch" unless official reproductions are included.
2. "Depth improves segmentation" as the central novelty.
3. "First depth-based SSL segmentation" unless the claim is heavily qualified.
Use:
"To our knowledge, GeoRisk-SPC is the first framework to use monocular depth discontinuity as a pseudo-label risk prior for semi-supervised surgical instrument segmentation."
## Recommended Main-Paper Tables And Figures
### Table 1: Controlled Baseline Comparison
Rows:
Sup-only, MT, UAMT, URPC, CPS, BCP or MC-Net+, UniMatch official, MMS, SegMatch official, GeoRisk-SPC, GeoRisk-SPC+DGv4.
Columns:
5%, 10%, 20%, 30%, 40% Dice and IoU. If space is tight, put Dice in main text and IoU/boundary metrics in appendix.
### Table 2: Depth Fairness
Rows:
Sup-only RGB, Sup-only RGB-D, MT RGB, MT RGB-D, UniMatch RGB, UniMatch RGB-D, SegMatch RGB, SegMatch RGB-D, GeoRisk risk-only, GeoRisk RGB-D.
Columns:
10% and 20% Dice, boundary F1, NSD/HD95.
### Table 3: Ablation
Rows:
risk components, supervision strategy, perturbation type, boundary loss, threshold variant.
Columns:
Dice, boundary F1, risk-error AUROC.
### Table 4: Robustness And Generalization
Rows:
EndoVis 2017, EndoVis 2018, Robust-MIS/CholecInstanceSeg if available.
Columns:
Dice, IoU, boundary F1, HD95.
### Figure 1: Method
Show weak teacher pseudo-label, depth discontinuity, uncertainty, conflict, risk map, low-risk hard supervision, high-risk SPC.
### Figure 2: Risk Validity
Show risk map overlaid with actual pseudo-label error map. Include scatter or reliability plot showing risk score versus pseudo-label error probability.
### Figure 3: Qualitative Comparison
Compare MT, UniMatch official, SegMatch official, GeoRisk-SPC on boundary drift, thin instrument shaft, specular highlight, occlusion, and false merging.
### Figure 4: Label Efficiency
Plot Dice and boundary F1 versus label ratio with CIs.
## Submission-Framing Guidance
### For MICCAI 2026
Pitch:
GeoRisk-SPC is a surgical-domain SSL method that introduces geometry-conditioned pseudo-label risk localization and region-adaptive structural consistency. The novelty is targeted and appropriate for MICCAI if the experiments show risk maps correspond to actual pseudo-label failures and the method beats official surgical SSL baselines.
Risk:
MICCAI reviewers will reject if the method looks like Mean Teacher + depth + feature perturbation without strong ablations.
### For TMI 2026
Pitch:
GeoRisk-SPC is a clinically motivated, geometry-aware semi-supervised framework for label-efficient surgical instrument segmentation, validated across datasets, label budgets, depth estimators, and structural error modes.
Risk:
TMI reviewers will expect broader validation, statistical rigor, and comparison with recent surgical and medical SSL methods. One dataset and simplified baselines are not enough.
## Immediate Action Plan
### P0 Runs
1. Finish UniMatch official on 5/10/20/40.
2. Finish SegMatch official on 10/30 first, then 5/20/40.
3. Add MMS on 5/10/20/40.
4. Add CPS to main table.
5. Add one of BCP or MC-Net+.
6. Run depth controls for MT, UniMatch, and SegMatch at 10% and 20%.
7. Run risk-map component ablations at 10% and 20%.
### P1 Runs
1. Add CrossMatch if structural perturbation novelty is challenged.
2. Add UTMT if you expand to SAR-RARP50/CaDIS or want a recent surgical endoscopic SSL comparator.
3. Add DiffRect only as appendix or related work unless implementation is cheap.
4. Add EndoVis 2018 binary collapsed.
5. Run three seeds for top methods at 10% and 20%.
### Paper Edits
1. Remove or rename simplified UniMatch/SegMatch rows unless official versions are added.
2. Replace "competitive with recent strong baselines" with precise wording until official results exist.
3. Clarify the EndoVis 2017 split protocol.
4. Add statistical reporting plan.
5. Add a risk-validity analysis section.
6. Add a depth fairness paragraph in limitations or experiments.
## Key Sources
1. SegMatch: semi-supervised surgical instrument segmentation, Scientific Reports 2025: https://www.nature.com/articles/s41598-025-94568-z
2. Min-Max Similarity: A Contrastive Semi-Supervised Deep Learning Network for Surgical Tools Segmentation, IEEE TMI 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10597739/
3. Surgivisor: Transformer-based semi-supervised instrument segmentation for endoscopic surgery, Biomedical Signal Processing and Control 2024: https://www.sciencedirect.com/science/article/abs/pii/S1746809423008674
4. UniMatch: Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation, CVPR 2023: https://arxiv.org/abs/2208.09910
5. UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation, TPAMI 2025: https://arxiv.org/abs/2410.10777
6. BCP: Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
7. DiffRect: Latent Diffusion Label Rectification for Semi-supervised Medical Image Segmentation, MICCAI 2024: https://papers.miccai.org/miccai-2024/222-Paper3391.html
8. CrossMatch: Enhance Semi-Supervised Medical Image Segmentation with Perturbation Strategies and Knowledge Distillation, 2024: https://arxiv.org/abs/2405.00354
9. CorrMatch: Label Propagation via Correlation Matching for Semi-Supervised Semantic Segmentation, CVPR 2024: https://arxiv.org/abs/2306.04300
10. MC-Net+: Mutual consistency learning for semi-supervised medical image segmentation, Medical Image Analysis 2022: https://www.sciencedirect.com/science/article/pii/S1361841522001773
11. UTMT: Semi-supervised segmentation of surgical endoscopic images based on Uncertainty guided Twin Mean Teacher, Biomedical Signal Processing and Control 2026: https://www.sciencedirect.com/science/article/pii/S1746809425012467
12. URPC: Semi-supervised medical image segmentation via uncertainty rectified pyramid consistency, Medical Image Analysis 2022: https://pubmed.ncbi.nlm.nih.gov/35732106/
13. EndoVis 2017 Robotic Instrument Segmentation Challenge: https://arxiv.org/abs/1902.06426
14. EndoVis 2018 Robotic Scene Segmentation Challenge: https://arxiv.org/abs/2001.11190
