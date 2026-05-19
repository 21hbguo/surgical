import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        sim = sim[~mask].view(sim.shape[0], -1)
        labels = torch.arange(z_i.shape[0], device=z_i.device)
        labels = torch.cat([labels + z_i.shape[0] - 1, labels], dim=0)
        return F.cross_entropy(sim, labels)


class CoordLoss(nn.Module):
    def __init__(self, conf_thresh=0.5, ignore_background=True):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.ignore_background = ignore_background
        self.eps = 1e-6

    def forward(self, output, label):
        _, C, _, _ = output.shape
        prob = F.softmax(output, dim=1)
        label_one_hot = F.one_hot(label.long(), C).permute(0, 3, 1, 2).float()

        if self.ignore_background:
            if C <= 1:
                return output.new_tensor(0.0)
            prob = prob[:, 1:, ...]
            label_one_hot = label_one_hot[:, 1:, ...]

        h_cdf_pred = torch.cumsum(prob, dim=2)
        h_cdf_gt = torch.cumsum(label_one_hot, dim=2)
        w_cdf_pred = torch.cumsum(prob, dim=3)
        w_cdf_gt = torch.cumsum(label_one_hot, dim=3)

        mass_pred = prob.sum(dim=(2, 3), keepdim=True) + self.eps
        mass_gt = label_one_hot.sum(dim=(2, 3), keepdim=True) + self.eps

        h_cdf_pred = h_cdf_pred / mass_pred
        h_cdf_gt = h_cdf_gt / mass_gt
        w_cdf_pred = w_cdf_pred / mass_pred
        w_cdf_gt = w_cdf_gt / mass_gt

        loss_h = torch.abs(h_cdf_pred - h_cdf_gt).mean(dim=(2, 3))
        loss_w = torch.abs(w_cdf_pred - w_cdf_gt).mean(dim=(2, 3))
        loss_per_class = 0.5 * (loss_h + loss_w)

        gt_active = (label_one_hot.sum(dim=(2, 3)) > 0).float()
        if gt_active.sum() == 0:
            return output.new_tensor(0.0)
        loss = (loss_per_class * gt_active).sum() / (gt_active.sum() + self.eps)
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = 1e-5

    def _one_hot_encoder(self, input_tensor):
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.squeeze(1)
        return (
            F.one_hot(input_tensor.long(), num_classes=self.n_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size()

        inputs_flat = inputs.reshape(-1, self.n_classes)
        target_flat = target.reshape(-1, self.n_classes)

        intersection = (inputs_flat * target_flat).sum(0)
        cardinality = inputs_flat.sum(0) + target_flat.sum(0)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
    return (input_softmax - target_softmax) ** 2


def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    ratio = min_beta / max_beta
    exponent = epoch / total_epochs
    beta = max_beta * (ratio**exponent)
    return beta


def sigmoid_rampup(
    current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0
):
    if total_rampup_epochs == 0:
        return max_threshold
    current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
    phase = 1.0 - (current_epoch / total_rampup_epochs)
    ramp = math.exp(-steepness * (phase**2))
    return min_threshold + (max_threshold - min_threshold) * ramp


class UnCLoss(nn.Module):
    def __init__(self):
        super(UnCLoss, self).__init__()

    def forward(self, s_logits, t_logits, beta):
        EPS = 1e-6
        p_s = F.softmax(s_logits, dim=1)
        p_s_log = torch.log(p_s + EPS)
        H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)
        p_t = F.softmax(t_logits, dim=1)
        p_t_log = torch.log(p_t + EPS)
        H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)
        loss = (p_s - p_t) ** 2 / (exp_H_s + exp_H_t)
        loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))
        return loss.mean()


class FeCLoss(nn.Module):
    def __init__(
        self,
        device,
        temperature=0.6,
        gamma=2.0,
        use_focal=False,
        rampup_epochs=2000,
        lambda_cross=1.0,
    ):
        super(FeCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(
        self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0
    ):
        B, N, _ = feat.shape
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()
        mem_mask_neg = 1 - mem_mask
        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature
        identity = torch.eye(N, device=self.device)
        neg_identity = 1 - identity
        feat_logits = feat_logits * neg_identity
        feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
        feat_logits = feat_logits - feat_logits_max.detach()
        exp_logits = torch.exp(feat_logits)
        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)
        denominator = exp_logits + neg_sum.unsqueeze(dim=-1)
        division = exp_logits / (denominator + 1e-18)
        loss_matrix = -torch.log(division + 1e-18)
        loss_matrix = loss_matrix * mem_mask * neg_identity
        loss_student = torch.sum(loss_matrix, dim=-1) / (
            torch.sum(mem_mask, dim=-1) - 1 + 1e-18
        )
        loss_student = loss_student.mean()
        if self.use_focal:
            similarity = division
            focal_weights = torch.ones_like(similarity)
            pos_thresh = sigmoid_rampup(
                epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5
            )
            neg_thresh = sigmoid_rampup(
                epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5
            )
            hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
            focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(
                self.gamma
            )
            hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)
            focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)
            loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / (
                torch.sum(mem_mask, dim=-1) - 1 + 1e-18
            )
            loss_student = loss_student.mean()
        if gambling_uncertainty is not None:
            loss_student_per_patch = torch.sum(loss_matrix, dim=-1) / (
                torch.sum(mem_mask, dim=-1) - 1 + 1e-18
            )
            loss_student = (loss_student_per_patch * gambling_uncertainty).mean()
        loss_cross = 0.0
        if teacher_feat is not None:
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2))
            mem_mask_cross = torch.eq(mask, mask.transpose(1, 2)).float()
            mem_mask_cross_neg = 1 - mem_mask_cross
            cross_neg_thresh = sigmoid_rampup(
                epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5
            )
            cross_hard_neg_mask = mem_mask_cross_neg.bool() & (
                cross_sim > cross_neg_thresh
            )
            if cross_hard_neg_mask.sum() > 0:
                loss_cross_term = -torch.log(1 - cross_sim + 1e-18)
                loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
                loss_cross = torch.sum(loss_cross_term) / (
                    torch.sum(cross_hard_neg_mask.float()) + 1e-18
                )
            else:
                loss_cross = 0.0
        total_loss = loss_student + self.lambda_cross * loss_cross
        return total_loss


class _L1Loss(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).mean()


class _GradientLoss(nn.Module):
    def forward(self, pred, target):
        pred_dx = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        pred_dy = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        target_dx = torch.abs(target[..., 1:, :] - target[..., :-1, :])
        target_dy = torch.abs(target[..., :, 1:] - target[..., :, :-1])
        return (
            torch.abs(pred_dx - target_dx).mean()
            + torch.abs(pred_dy - target_dy).mean()
        )


class _SmoothnessLoss(nn.Module):
    def forward(self, pred, target=None):
        pred_dx = torch.abs(pred[..., 1:, :] - pred[..., :-1, :])
        pred_dy = torch.abs(pred[..., :, 1:] - pred[..., :, :-1])
        return pred_dx.mean() + pred_dy.mean()


class _SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.window = None

    def _gaussian_window(self, window_size, channels):
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / (2.0 * 1.5**2))
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        channels = pred.shape[1]
        if self.window is None or self.window.shape[0] != channels:
            self.window = self._gaussian_window(self.window_size, channels).to(
                pred.device
            )
        window = self.window

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                pred * pred, window, padding=self.window_size // 2, groups=channels
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                target * target, window, padding=self.window_size // 2, groups=channels
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                pred * target, window, padding=self.window_size // 2, groups=channels
            )
            - mu1_mu2
        )

        SSIM = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return 1 - SSIM.mean()


class _RangeLoss(nn.Module):
    def forward(self, pred, min_val=0.0, max_val=1.0):
        return torch.relu(min_val - pred).mean() + torch.relu(pred - max_val).mean()


class DepthLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        gradient_weight=0.1,
        smoothness_weight=0.01,
        ssim_weight=0.5,
        range_weight=0.1,
    ):
        super().__init__()
        self.l1_loss = _L1Loss()
        self.gradient_loss = _GradientLoss()
        self.smoothness_loss = _SmoothnessLoss()
        self.ssim_loss = _SSIMLoss()
        self.range_loss = _RangeLoss()

        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight
        self.ssim_weight = ssim_weight
        self.range_weight = range_weight

    def forward(self, pred, target):
        loss_dict = {
            "depth_l1": self.l1_loss(pred, target),
            "depth_gradient": self.gradient_loss(pred, target),
            "depth_smoothness": self.smoothness_loss(pred),
            "depth_ssim": self.ssim_loss(pred, target),
            "depth_range": self.range_loss(pred),
        }
        weighted_loss = (
            self.l1_weight * loss_dict["depth_l1"]
            + self.gradient_weight * loss_dict["depth_gradient"]
            + self.smoothness_weight * loss_dict["depth_smoothness"]
            + self.ssim_weight * loss_dict["depth_ssim"]
            + self.range_weight * loss_dict["depth_range"]
        )
        loss_dict["depth_total"] = weighted_loss
        return loss_dict


class CosineSimilarityContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, temperature=0.07):
        super(CosineSimilarityContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, anchor, positive, negative=None):
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)

        pos_sim = F.cosine_similarity(anchor_norm, positive_norm, dim=1)

        if negative is not None:
            negative_norm = F.normalize(negative, p=2, dim=-1)

            if negative_norm.dim() == 3:
                anchor_norm_exp = anchor_norm.unsqueeze(1)
                neg_sim = F.cosine_similarity(anchor_norm_exp, negative_norm, dim=2)
                neg_sim = neg_sim.max(dim=1)[0]
            else:
                if anchor_norm.size(0) != negative_norm.size(0):
                    neg_sim_list = []
                    for i in range(anchor_norm.size(0)):
                        sim = F.cosine_similarity(
                            anchor_norm[i : i + 1], negative_norm, dim=1
                        )
                        neg_sim_list.append(sim.max())
                    neg_sim = torch.stack(neg_sim_list)
                else:
                    neg_sim = F.cosine_similarity(anchor_norm, negative_norm, dim=1)

            losses = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
            loss = losses.mean()

            return loss, pos_sim.mean(), neg_sim.mean()

        else:
            batch_size = anchor_norm.size(0)
            similarity_matrix = (
                torch.matmul(anchor_norm, positive_norm.transpose(0, 1))
                / self.temperature
            )
            labels = torch.arange(batch_size).to(anchor.device)
            loss = F.cross_entropy(similarity_matrix, labels)

            return loss, pos_sim.mean(), torch.tensor(0.0, device=anchor.device)
