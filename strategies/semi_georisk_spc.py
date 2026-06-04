"""GeoRisk-SPC: Geometry-aware Risk Localization + Structural Perturbation Consistency.

Uses depth discontinuity as structural risk prior, combined with teacher uncertainty
and geometry-semantic conflict, to locate high-risk pseudo-label regions.
Low-risk: hard pseudo-label supervision.
High-risk: feature perturbation + soft consistency + boundary consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import BaseTrainingStrategy


def _local_relative_normalization(depth, window=16):
    """Local mean/std normalization to remove global depth bias."""
    pad = window // 2
    depth_pad = F.pad(depth, [pad] * 4, mode='reflect')
    B, C, H, W = depth.shape
    patches = depth_pad.unfold(2, window, 1).unfold(3, window, 1)
    local_mean = patches.mean(dim=(-1, -2), keepdim=False)
    local_std = patches.std(dim=(-1, -2), keepdim=False) + 1e-6
    local_mean = local_mean[:, :, :H, :W]
    local_std = local_std[:, :, :H, :W]
    return (depth - local_mean) / local_std


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


def _sobel_gradient_magnitude_single(x):
    """Compute gradient magnitude for single-channel input: [B, 1, H, W]."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _normalize_map(x):
    """Min-max normalize to [0, 1]."""
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + 1e-8)


class GeoRiskSPCStrategy(BaseTrainingStrategy):

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
        self.risk_source = getattr(args, 'risk_source', 'all')
        self.risk_no_supervision = getattr(args, 'risk_no_supervision', False)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def _compute_risk_map(self, depth, teacher_pred):
        """Compute geometry-aware risk map from depth and teacher predictions.

        Returns:
            M_r: [B_u, 1, H, W] high-risk mask
            M_l: [B_u, 1, H, W] low-risk (high-confidence) mask
        """
        # 1. Local relative depth normalization
        d_rel = _local_relative_normalization(depth, self.window_size)

        # 2. Depth discontinuity [B, 1, H, W]
        G_d = _sobel_gradient_magnitude_single(d_rel)

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

        return M_r, M_l

    def _compute_boundary_gradients(self, pred):
        """Compute spatial gradient magnitude of prediction maps."""
        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B * C, 1, H, W)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        gx = F.conv2d(pred_flat, sobel_x, padding=1)
        gy = F.conv2d(pred_flat, sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.reshape(B, C, H, W)

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
        risk_mask = None
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
                    risk_mask, M_l = self._compute_risk_map(unlabeled_depth, teacher_pred)
        else:
            teacher_pred = None

        if teacher_pred is not None:
            batch_data['teacher_pred'] = teacher_pred

        # Student input: always concatenate depth if available
        stud_volume = torch.cat([stud_image, depth_tensor], dim=1) if depth_tensor is not None else stud_image

        # Build full-batch risk mask (zeros for labeled, actual mask for unlabeled)
        if risk_mask is not None:
            B_full = stud_volume.shape[0]
            full_risk_mask = torch.zeros(B_full, 1, *risk_mask.shape[2:], device=self.device)
            full_risk_mask[self.labeled_bs:] = risk_mask
            output_clean, output_pert = self.model(stud_volume, risk_mask=full_risk_mask)
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
        pl_loss = torch.tensor(0.0, device=self.device)
        cons_loss = torch.tensor(0.0, device=self.device)
        bd_loss = torch.tensor(0.0, device=self.device)
        consistency_weight = self._get_consistency_weight(iter_num)

        if iter_num >= self.consistency_start_iters and B_u > 0 and teacher_pred is not None:
            clean_unlabeled = output_clean[self.labeled_bs:]
            clean_unlabeled_soft = clean_soft[self.labeled_bs:]
            pseudo_label = teacher_pred.argmax(dim=1)

            if self.risk_no_supervision:
                # Standard semi-supervised: all unlabeled pixels get pseudo-label loss
                pl_ce = F.cross_entropy(clean_unlabeled, pseudo_label.long(), reduction='none')
                pl_loss = pl_ce.mean()
            elif risk_mask is not None and M_l is not None and output_pert is not None:
                # Resize masks to prediction spatial size
                H, W = clean_unlabeled.shape[2:]
                M_r_up = F.interpolate(risk_mask, size=(H, W), mode='nearest')
                M_l_up = F.interpolate(M_l, size=(H, W), mode='nearest')

                # L_pl: low-risk pseudo-label loss
                pl_ce = F.cross_entropy(clean_unlabeled, pseudo_label.long(), reduction='none')
                pl_pixels = M_l_up.sum()
                if pl_pixels > 0:
                    pl_loss = (M_l_up.squeeze(1) * pl_ce).sum() / (pl_pixels + 1e-8)

                # L_cons: high-risk soft consistency (KL divergence)
                pert_unlabeled_soft = torch.softmax(output_pert[self.labeled_bs:], dim=1)
                kl = self.kl_loss(
                    torch.log(pert_unlabeled_soft + 1e-8),
                    clean_unlabeled_soft.detach()
                ).sum(dim=1, keepdim=True)
                r_pixels = M_r_up.sum()
                if r_pixels > 0:
                    cons_loss = (M_r_up * kl).sum() / (r_pixels + 1e-8)

                # L_bd: boundary consistency
                grad_clean = self._compute_boundary_gradients(clean_unlabeled_soft.detach())
                grad_pert = self._compute_boundary_gradients(pert_unlabeled_soft)
                bd_diff = torch.abs(grad_clean - grad_pert)
                if r_pixels > 0:
                    bd_loss = (M_r_up * bd_diff).sum() / (r_pixels + 1e-8)

        total_loss = (supervised_loss
                      + consistency_weight * self.pl_weight * pl_loss
                      + consistency_weight * self.cons_weight * cons_loss
                      + consistency_weight * self.bd_weight * bd_loss)

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'pl': pl_loss,
            'consistency': cons_loss,
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
        with torch.no_grad():
            volume = batch_data['image'].to(self.device)
            depth_tensor = self._get_depth_tensor(batch_data)
            if depth_tensor is not None:
                volume = torch.cat([volume, depth_tensor], dim=1)
            output = self.model(volume)
            if isinstance(output, tuple):
                output = output[0]
            return output
