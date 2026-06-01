# 半监督基线复现审计
## 结论
当前 `semi_unimatch.py` 和 `semi_segmatch.py` 不能作为 UniMatch、SegMatch 复现。它们实现的是 EMA teacher + 置信度伪标签 + 高斯噪声/简化 I-FGSM 的 FixMatch/Mean-Teacher 变体。
## P0 问题
1. `semi_unimatch.py` 使用 `ema_model` 生成无标注伪标签，只对一个 student 强视图计算 CE；官方 UniMatch 主逻辑使用同一模型 weak 预测、两个 strong 分支、两个 CutMix mask、weak feature perturbation 分支。
2. `semi_unimatch.py` 没有 CutMix、没有 `img_u_s1/img_u_s2`、没有 `pred_u_w_fp`、没有 `0.25/0.25/0.5` 无监督损失加权。
3. `semi_segmatch.py` 默认 `adversarial_eps=0.01`、`adversarial_steps=1`，且每步直接加 `eps`，再投影到 `[-2eps,2eps]`；这不是 `eps=0.08,K=25` 的 I-FGSM。
4. `semi_segmatch.py` 没有空间弱增强参数记录、没有 teacher 预测逆变换回 student 坐标系，无法复现空间 weak-to-strong 伪标签对齐。
5. `core/train.py` 当前只构造普通 `RandomGenerator` 样本，策略只收到 `image/label`，没有 weak/strong/CutMix 所需字段。
## P1 问题
1. 当前 UniMatch/SegMatch 名称污染实验表，论文中不能写作 UniMatch 或 SegMatch。
2. 当前损失归一化按 confident mask 数量除，官方 UniMatch 按 valid ignore mask 数量除，低置信像素作为 0 loss 保留分母，损失尺度不同。
3. CPS 结构接近官方双网络 CPS，但优化器、rampup、是否延迟一致性启动仍需与官方脚本对齐。
## 修改计划
1. 保留当前简化版，重命名为 `ema_fixmatch_noise` 和 `ema_fixmatch_ifgsm_lite`。
2. 新增 `UniMatchOfficialStrategy`，输入字段包含 `image_w/image_s1/image_s2/cutmix_box1/cutmix_box2`，实现双强分支、CutMix 伪标签替换、feature perturbation logits。
3. 新增 `SegMatchOfficialStrategy`，实现空间弱增强、逆变换、`eps=0.08,K=25` 的投影 I-FGSM。
4. 修改 `data/transforms.py` 和 `core/train.py`，使特定 strategy 使用对应半监督增强 transform，而不是只在 strategy 内部加高斯噪声。
5. 修改 `strategies/specs.py`，让 `unimatch` 和 `segmatch` 指向官方复现，旧实现只作为 lite/ablation alias。
