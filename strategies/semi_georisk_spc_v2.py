"""GeoRisk-SPC: Geometry-aware Risk Localization + Structural Perturbation Consistency.

Uses depth discontinuity as structural risk prior, combined with teacher uncertainty
and geometry-semantic conflict, to locate high-risk pseudo-label regions.
Low-risk: hard pseudo-label supervision.
High-risk: feature perturbation + soft consistency + boundary consistency.
"""

import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy

def _local_relative_normalization(depth, window=16):
    """Fast version using conv2d instead of unfold. GPU 3-5x faster, numerically equivalent."""
    pad = window // 2
    C = depth.size(1)
    depth_pad = F.pad(depth, [pad] * 4, mode='reflect')
    kernel = torch.ones((C, 1, window, window), device=depth.device, dtype=depth.dtype) / (window * window)
    local_mean = F.conv2d(depth_pad, kernel, groups=C)
    depth_sq = depth_pad ** 2
    mean_sq = F.conv2d(depth_sq, kernel, groups=C)
    N = window * window
    local_var = (mean_sq - local_mean ** 2) * N / (N - 1)
    local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))
    H, W = depth.shape[2], depth.shape[3]
    return (depth - local_mean[:, :, :H, :W]) / local_std[:, :, :H, :W]

def _sobel_gradient_magnitude(x):
    """Compute gradient magnitude using Sobel filters. Input: [B, C, H, W]."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    B, C, H, W = x.shape
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, sobel_x, padding=1)
    gy = F.conv2d(x_flat, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return mag.reshape(B, C, H, W)


def _normalize_map(x):
    """Min-max normalize to [0, 1]."""
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + 1e-8)


class GeoRiskSPCStrategyV2(BaseTrainingStrategy):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--risk_window_size', type=int, default=16)
        parser.add_argument('--risk_tau_r', type=float, default=0.5,
                            help='High-risk threshold')
        parser.add_argument('--risk_tau_c', type=float, default=0.9,
                            help='High-confidence threshold')
        parser.add_argument('--risk_lambda_c', type=float, default=0.5,
                            help='Conflict term weight')
        parser.add_argument('--risk_dropout_rate', type=float, default=0.3)
        parser.add_argument('--risk_noise_std', type=float, default=0.1)
        parser.add_argument('--risk_pl_weight', type=float, default=1.0,
                            help='Low-risk pseudo-label loss weight')
        parser.add_argument('--risk_cons_weight', type=float, default=1.0,
                            help='High-risk consistency loss weight')
        parser.add_argument('--risk_bd_weight', type=float, default=0.5,
                            help='Boundary consistency loss weight')
        parser.add_argument('--risk_source', type=str, default='all',
                            choices=['all', 'depth', 'uncertainty', 'conflict', 'depth_uncertainty'],
                            help='Risk sources: all=U_t*G_d+lambda*C, depth=G_d only, uncertainty=U_t only, conflict=C only, depth_uncertainty=U_t*G_d')
        parser.add_argument('--risk_no_supervision', action='store_true',
                            help='Disable risk-based supervision (standard semi-supervised)')

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        if int(args.use_depth or 0) not in (1, 13):
            raise ValueError("GeoRisk-SPC requires --use_depth 1 or 13.")
        self._enable_ema_support()
        self.consistency_start_iters = int(args.consistency_start_iters)
        self.window_size = args.risk_window_size
        self.tau_r = args.risk_tau_r
        self.tau_c = args.risk_tau_c
        self.lambda_c = args.risk_lambda_c
        self.pl_weight = args.risk_pl_weight
        self.cons_weight = args.risk_cons_weight
        self.bd_weight = args.risk_bd_weight
        self.risk_source = args.risk_source
        self.risk_no_supervision = args.risk_no_supervision

    def _compute_risk_map_v2(self, depth, teacher_pred, iter_num=0):
        """Compute geometry-aware risk map from depth and teacher predictions.

        Returns:
            M_r: [B_u, 1, H, W] high-risk mask
            M_l: [B_u, 1, H, W] low-risk (high-confidence) mask
        """
        U_t = -torch.sum(teacher_pred * torch.log(teacher_pred + 1e-8), dim=1, keepdim=True)
        U_t_norm = _normalize_map(U_t)

        M_r = (U_t_norm > 0.5).float()
        M_l = 1.0 - M_r
        if torch.rand(1).item() < 0.01:
            save_dir = '/home/guo/project/ssl4mis/data/test'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(M_r.shape[0]):
                vars_list = [
                    ('U_t_norm', U_t_norm[i, 0]),
                    ('M_r', M_r[i, 0]),
                    ('M_l', M_l[i, 0]),
                ]
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                for ax, (name, tensor) in zip(axes.flat, vars_list):
                    ax.imshow(tensor.cpu().numpy(), cmap='hot')
                    ax.set_title(name)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'risk_batch{iter_num}_sample{i}.png'), dpi=100, bbox_inches='tight')
                plt.close(fig)

        return M_r, M_l

    def _compute_risk_map_v1(self, depth, teacher_pred, iter_num=0):
        """Compute geometry-aware risk map from depth and teacher predictions.

        Returns:
            M_r: [B_u, 1, H, W] high-risk mask
            M_l: [B_u, 1, H, W] low-risk (high-confidence) mask
        """
        # 1. Local relative depth normalization
        d_rel = _local_relative_normalization(depth, self.window_size)

        # 2. Depth discontinuity [B, 1, H, W]
        G_d = _sobel_gradient_magnitude(d_rel)

        # 3. Teacher uncertainty (prediction entropy) [B, 1, H, W]
        U_t = -torch.sum(teacher_pred * torch.log(teacher_pred + 1e-8), dim=1, keepdim=True)

        # 4. Prediction boundary map - reduce to single channel via mean
        B_p = _sobel_gradient_magnitude(teacher_pred)  # [B, C, H, W]
        B_p = B_p.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 5. Geometry-semantic conflict
        G_d_norm = _normalize_map(G_d)
        B_p_norm = _normalize_map(B_p)
        C_conf = torch.abs(G_d_norm - B_p_norm)

        # 6. Final risk map (selectable sources)
        U_t_norm = _normalize_map(U_t)
        if self.risk_source == 'depth':
            R = G_d_norm
        elif self.risk_source == 'uncertainty':
            R = U_t_norm
        elif self.risk_source == 'conflict':
            R = C_conf
        elif self.risk_source == 'depth_uncertainty':
            R = U_t_norm * G_d_norm
        else:  # 'all'
            R = U_t_norm * G_d_norm + self.lambda_c * C_conf

        # 7. Region masks
        conf = teacher_pred.max(dim=1, keepdim=True)[0]
        M_r = (R > self.tau_r).float()
        M_l = (conf > self.tau_c).float() * (1.0 - M_r)

        # Save risk map intermediates (1% prob, one image per sample)
        if torch.rand(1).item() < 0.01:
            save_dir = '/home/guo/project/ssl4mis/data/test'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(M_r.shape[0]):
                vars_list = [
                    ('depth', depth[i, 0]),
                    ('d_rel', d_rel[i, 0]),
                    ('G_d_norm', G_d_norm[i, 0]),
                    ('U_t_norm', U_t_norm[i, 0]),
                    ('U_t*G_d', U_t_norm[i, 0] * G_d_norm[i, 0]),
                    ('B_p_norm', B_p_norm[i, 0]),
                    ('C_conf', C_conf[i, 0]),
                    ('R', R[i, 0]),
                    ('M_r', M_r[i, 0]),
                    ('M_l', M_l[i, 0]),
                ]
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                for ax, (name, tensor) in zip(axes.flat, vars_list):
                    ax.imshow(tensor.cpu().numpy(), cmap='hot')
                    ax.set_title(name)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'risk_batch{iter_num}_sample{i}.png'), dpi=100, bbox_inches='tight')
                plt.close(fig)

        return M_r, M_l

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        image = batch_data['image'].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        label = batch_data['label'].to(self.device)

        # Strong augmentation for student
        stud_image = self._add_noise(image, strong_flag='s', unlabeled_only=True)

        # Split labeled / unlabeled
        unlabeled_image = image[self.labeled_bs:]
        unlabeled_depth = depth_tensor[self.labeled_bs:] if depth_tensor is not None else None
        B_u = unlabeled_image.shape[0]

        # Risk mask placeholder
        M_r = None
        M_l = None

        # Teacher forward + risk map (only when unlabeled data exists)
        if B_u > 0:
            with torch.no_grad():
                ema_inputs = self._add_noise(unlabeled_image, strong_flag='t')
                if unlabeled_depth is not None:
                    ema_inputs = torch.cat([ema_inputs, unlabeled_depth], dim=1)
                ema_output = self.ema_model(ema_inputs)
                if isinstance(ema_output, tuple):
                    ema_output = ema_output[0]
                teacher_pred = torch.softmax(ema_output, dim=1)

                # Compute risk map
                if unlabeled_depth is not None:
                    M_r, M_l = self._compute_risk_map_v2(unlabeled_depth, teacher_pred, iter_num)
                    
        else:
            teacher_pred = None

        if teacher_pred is not None:
            batch_data['teacher_pred'] = teacher_pred

        # Student input: always concatenate depth if available
        stud_volume = torch.cat([stud_image, depth_tensor], dim=1) if depth_tensor is not None else stud_image

        # Build full-batch risk mask (zeros for labeled, actual mask for unlabeled)
        if M_r is not None:
            full_M_r = torch.zeros(stud_volume.shape[0], 1, *M_r.shape[2:], device=self.device)
            full_M_r[self.labeled_bs:] = M_r
            output_clean, output_pert = self.model(stud_volume, M_r=full_M_r)
        else:
            output_clean = self.model(stud_volume)
            output_pert = None

        if isinstance(output_clean, tuple):
            output_clean = output_clean[0]
        if isinstance(output_pert, tuple):
            output_pert = output_pert[0]

        clean_soft = torch.softmax(output_clean, dim=1)

        # === Supervised loss (labeled data) ===
        loss_ce = self.ce_loss(output_clean[:self.labeled_bs], label[:self.labeled_bs].long())
        loss_dice = self.dice_loss(clean_soft[:self.labeled_bs], label[:self.labeled_bs].unsqueeze(1))
        supervised_loss = 0.5 * (loss_dice + loss_ce)

        # === Unlabeled losses ===
        cons_loss = torch.tensor(0.0, device=self.device)
        clean_teacher_consistency = torch.tensor(0.0, device=self.device)
        pert_teacher_consistency = torch.tensor(0.0, device=self.device)
        bd_loss = torch.tensor(0.0, device=self.device)
        consistency_weight = self._get_consistency_weight(iter_num)

        clean_unlabeled_soft = clean_soft[self.labeled_bs:]

        if iter_num >= self.consistency_start_iters and B_u > 0 and teacher_pred is not None:
            clean_teacher_consistency = torch.mean((clean_unlabeled_soft - teacher_pred) ** 2)

            if M_r is not None and M_l is not None and output_pert is not None:
                pert_unlabeled_soft = torch.softmax(output_pert[self.labeled_bs:], dim=1)
                pert_teacher_consistency = torch.mean((pert_unlabeled_soft - teacher_pred) ** 2)

            cons_loss = clean_teacher_consistency + pert_teacher_consistency

        total_loss = (supervised_loss
                      + consistency_weight * self.cons_weight * cons_loss)

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': cons_loss,
            'clean_teacher_consistency': clean_teacher_consistency,
            'pert_teacher_consistency': pert_teacher_consistency,
            'boundary': bd_loss,
            'consistency_weight': consistency_weight,
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict

    def validation_step(self, batch_data):
        volume = batch_data['image'].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        output = self.model(volume)
        if isinstance(output, tuple):
            output = output[0]
        return output
