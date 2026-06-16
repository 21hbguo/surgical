import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning (from U2PL)."""
    def __init__(self, in_channels, mid_channels=256, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class U2PLStrategy(BaseTrainingStrategy):
    """U2PL: Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels (CVPR 2022).

    Faithful reproduction based on official code:
    - EMA teacher generates pseudo-labels
    - Entropy-based percentile thresholding separates reliable/unreliable pixels
    - Unsupervised loss: CE on reliable pixels (dropping high-entropy ones)
    - Contrastive loss: memory bank with class-wise negative samples, cosine similarity
    - Projection head maps bottleneck features to contrastive space
    - Momentum prototype for positive anchors
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)
        self._enable_ema_support()

        num_classes = args.num_classes
        bottleneck_channels = 256  # filter_num * 16 for default UNet

        # U2PL hyperparameters (matching original config for binary segmentation)
        self.drop_percent = args.u2pl_drop_percent
        self.contrast_weight = args.u2pl_contrast_weight
        self.low_entropy_threshold = args.u2pl_low_entropy_threshold
        self.num_queries = args.u2pl_num_queries
        self.num_negatives = args.u2pl_num_negatives
        self.temperature = args.u2pl_temperature
        self.current_class_threshold = args.u2pl_class_threshold
        self.current_class_negative_threshold = args.u2pl_class_neg_threshold
        self.low_rank = args.u2pl_low_rank
        self.high_rank = args.u2pl_high_rank
        self.sup_only_epoch = args.u2pl_sup_only_epoch
        self.max_iterations = args.max_iterations

        # Projection head maps bottleneck features (256 channels) to contrastive space
        self.projector = ProjectionHead(bottleneck_channels, 256, 256).to(device)

        # Class-wise memory bank (from original U2PL)
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        for i in range(num_classes):
            self.memobank.append([torch.zeros(0, 256)])
            self.queue_size.append(30000)
            self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        self.queue_size[0] = 50000

        # Momentum prototype (from original U2PL)
        self.num_classes = num_classes
        self.prototype = torch.zeros((num_classes, self.num_queries, 1, 256), device=device)
        self.total_epochs = 1
        self.iters_per_epoch = None
        self.last_epoch = -1

        # Update optimizer to include projector
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001
        )

    def _extract_features(self, volume):
        """Extract logits and bottleneck features from model."""
        result = self.model(volume, return_features=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        # Fallback: extract features via encoder directly
        features = self.model.encoder(volume)
        logits = self.model.decoder(features)
        return logits, features[4]

    def _label_onehot(self, label, num_classes):
        B, H, W = label.shape
        onehot = torch.zeros(num_classes, B, H, W, device=label.device)
        label_tmp = label.clone()
        label_tmp[label == 255] = 0
        onehot.scatter_(0, label_tmp.unsqueeze(1), 1.0)
        onehot[:, label == 255] = 0
        return onehot.permute(1, 0, 2, 3)

    def _extract_ema_features(self, volume):
        result = self.ema_model(volume, return_features=True)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        features = self.ema_model.encoder(volume)
        logits = self.ema_model.decoder(features)
        return logits, features[4]

    def _dequeue_and_enqueue(self, keys, queue, queue_ptr, queue_size):
        keys = keys.detach()
        batch_size = keys.shape[0]
        if batch_size == 0:
            return 0
        keys_cpu = keys.cpu()
        queue[0] = torch.cat((queue[0], keys_cpu), dim=0)
        if queue[0].shape[0] >= queue_size:
            queue[0] = queue[0][-queue_size:, :]
            ptr = queue_size
        else:
            ptr = (int(queue_ptr[0].item()) + batch_size) % queue_size
        queue_ptr[0] = ptr
        return batch_size

    def _resolve_total_epochs(self, iter_num, epoch):
        if epoch != self.last_epoch:
            self.last_epoch = epoch
            if epoch > 0:
                self.iters_per_epoch = max(1, iter_num // epoch)
                self.total_epochs = max(epoch + 1, math.ceil(self.max_iterations / self.iters_per_epoch))
            else:
                self.total_epochs = max(1, self.total_epochs)
        return max(epoch + 1, self.total_epochs)

    def _compute_unsupervised_loss(self, pred, target, percent, pred_teacher):
        """Unsupervised loss with entropy-based percentile thresholding (from U2PL)."""
        batch_size, num_class, h, w = pred.shape

        with torch.no_grad():
            prob = torch.softmax(pred_teacher, dim=1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

            valid_mask = target != 255
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=pred.device)

            thresh = np.percentile(
                entropy[valid_mask].detach().cpu().numpy().flatten(), percent
            )
            thresh_mask = entropy.ge(thresh).bool() * valid_mask.bool()
            target = target.clone()
            target[thresh_mask] = 255
            weight = batch_size * h * w / torch.sum(target != 255).clamp(min=1)

        loss = weight * F.cross_entropy(pred, target.long(), ignore_index=255)
        return loss

    def _compute_contrastive_loss(self, rep, label_l, label_u, prob_l, prob_u, low_mask, high_mask, rep_teacher, i_iter=0):
        num_feat = rep.shape[1]
        num_segments = label_l.shape[1]

        low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
        high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

        rep = rep.permute(0, 2, 3, 1)
        rep_teacher = rep_teacher.permute(0, 2, 3, 1)

        _, prob_indices_l = torch.sort(prob_l, 1, True)
        prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)

        _, prob_indices_u = torch.sort(prob_u, 1, True)
        prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)

        prob = torch.cat((prob_l, prob_u), dim=0)

        seg_num_list = []
        seg_proto_list = []
        valid_classes = []
        seg_feat_low_entropy_list = []
        for i in range(num_segments):
            low_valid_pixel_seg = low_valid_pixel[:, i]
            high_valid_pixel_seg = high_valid_pixel[:, i]

            prob_seg = prob[:, i, :, :]
            rep_mask_low_entropy = (
                prob_seg > self.current_class_threshold
            ) * low_valid_pixel_seg.bool()
            rep_mask_high_entropy = (
                prob_seg < self.current_class_negative_threshold
            ) * high_valid_pixel_seg.bool()

            seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])
            seg_proto_list.append(torch.mean(rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True))
            class_mask_u = torch.sum(
                prob_indices_u[:, :, :, self.low_rank:self.high_rank].eq(i), dim=3
            ).bool()

            class_mask_l = torch.sum(
                prob_indices_l[:, :, :, :self.low_rank].eq(i), dim=3
            ).bool()

            class_mask = torch.cat(
                (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
            )
            negative_mask = rep_mask_high_entropy * class_mask

            keys = rep_teacher[negative_mask].detach()
            self._dequeue_and_enqueue(
                keys=keys,
                queue=self.memobank[i],
                queue_ptr=self.queue_ptrlis[i],
                queue_size=self.queue_size[i],
            )

            if low_valid_pixel_seg.sum() > 0:
                seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
                valid_classes.append(i)

        if len(seg_num_list) <= 1:
            return torch.tensor(0.0, device=rep.device) * rep.sum()

        reco_loss = torch.tensor(0.0, device=rep.device)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        prototype = torch.zeros((prob_indices_l.shape[-1], self.num_queries, 1, num_feat), device=rep.device)

        for i in range(valid_seg):
            cls_idx = valid_classes[i]
            if len(seg_feat_low_entropy_list[cls_idx]) == 0 or self.memobank[cls_idx][0].shape[0] == 0:
                reco_loss = reco_loss + 0 * rep.sum()
                continue
            seg_low_entropy_idx = torch.randint(len(seg_feat_low_entropy_list[cls_idx]), size=(self.num_queries,), device=rep.device)
            anchor_feat = seg_feat_low_entropy_list[cls_idx][seg_low_entropy_idx].clone()
            with torch.no_grad():
                negative_feat = self.memobank[cls_idx][0].clone().to(rep.device)
                high_entropy_idx = torch.randint(len(negative_feat), size=(self.num_queries * self.num_negatives,), device=rep.device)
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(self.num_queries, self.num_negatives, num_feat)
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                if not (self.prototype == 0).all():
                    ema_decay = min(1 - 1 / i_iter, 0.999)
                    positive_feat = (1 - ema_decay) * positive_feat + ema_decay * self.prototype[cls_idx]
                prototype[cls_idx] = positive_feat.clone()
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / self.temperature, torch.zeros(self.num_queries, dtype=torch.long, device=rep.device))
        self.prototype = prototype.detach()

        return reco_loss / valid_seg

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data["label"].to(self.device)

        labeled_volume = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_volume = volume[self.labeled_bs:]

        unsup_loss = torch.tensor(0.0, device=self.device)
        contra_loss = torch.tensor(0.0, device=self.device)

        if unlabeled_volume.shape[0] > 0 and epoch >= self.sup_only_epoch:
            total_epochs = self._resolve_total_epochs(iter_num, epoch)
            percent_unreliable = (100 - self.drop_percent) * (1 - epoch / total_epochs)
            drop_percent = 100 - percent_unreliable
            alpha_t = self.low_entropy_threshold * (1 - epoch / total_epochs)
            image_all = torch.cat((labeled_volume, unlabeled_volume), dim=0)
            num_labeled = labeled_volume.shape[0]
            pred_all, feat_all = self._extract_features(image_all)
            pred_l_large = pred_all[:num_labeled]
            pred_u_large = pred_all[num_labeled:]
            pred_l_large_soft = torch.softmax(pred_l_large, dim=1)
            loss_ce = self.ce_loss(pred_l_large, labeled_label.long())
            loss_dice = self.dice_loss(pred_l_large_soft, labeled_label.unsqueeze(1))
            sup_loss = 0.5 * (loss_dice + loss_ce)
            target_size = pred_u_large.shape[2:]
            feat_all_up = F.interpolate(feat_all, size=target_size, mode='bilinear', align_corners=False)
            rep_all = self.projector(feat_all_up)
            with torch.no_grad():
                pred_all_teacher, feat_all_teacher = self._extract_ema_features(image_all)
                prob_all_teacher = torch.softmax(pred_all_teacher, dim=1)
                prob_l_teacher = prob_all_teacher[:num_labeled]
                prob_u_teacher = prob_all_teacher[num_labeled:]
                feat_all_teacher_up = F.interpolate(feat_all_teacher, size=target_size, mode='bilinear', align_corners=False)
                rep_all_teacher = self.projector(feat_all_teacher_up.detach())
                pred_u_large_teacher = pred_all_teacher[num_labeled:]
                prob = torch.softmax(pred_u_large_teacher.detach(), dim=1)
                _, label_u_aug = torch.max(prob, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                valid_mask = label_u_aug != 255
                if valid_mask.sum() > 0:
                    low_thresh = np.percentile(entropy[valid_mask].cpu().numpy().flatten(), alpha_t)
                    low_entropy_mask = entropy.le(low_thresh).float() * valid_mask.bool()
                    high_thresh = np.percentile(entropy[valid_mask].cpu().numpy().flatten(), 100 - alpha_t)
                    high_entropy_mask = entropy.ge(high_thresh).float() * valid_mask.bool()
                else:
                    low_entropy_mask = torch.zeros_like(entropy)
                    high_entropy_mask = torch.zeros_like(entropy)
                low_mask_all = torch.cat(((labeled_label.unsqueeze(1) != 255).float(), low_entropy_mask.unsqueeze(1)))
                high_mask_all = torch.cat(((labeled_label.unsqueeze(1) != 255).float(), high_entropy_mask.unsqueeze(1)))
                target_h, target_w = rep_all.shape[2:]
                low_mask_all = F.interpolate(low_mask_all, size=(target_h, target_w), mode="nearest")
                high_mask_all = F.interpolate(high_mask_all, size=(target_h, target_w), mode="nearest")
                label_l_small = F.interpolate(self._label_onehot(labeled_label, self.num_classes), size=(target_h, target_w), mode="nearest")
                label_u_small = F.interpolate(self._label_onehot(label_u_aug, self.num_classes), size=(target_h, target_w), mode="nearest")
            unsup_loss = self._compute_unsupervised_loss(
                pred_u_large,
                label_u_aug.clone(),
                drop_percent,
                pred_u_large_teacher.detach()
            )

            contra_loss = self._compute_contrastive_loss(
                rep_all,
                label_l_small.long(),
                label_u_small.long(),
                prob_l_teacher.detach(),
                prob_u_teacher.detach(),
                low_mask_all,
                high_mask_all,
                rep_all_teacher.detach(),
                i_iter=max(1, iter_num),
            )
        else:
            output_labeled, _ = self._extract_features(labeled_volume)
            output_labeled_soft = torch.softmax(output_labeled, dim=1)
            loss_ce = self.ce_loss(output_labeled, labeled_label.long())
            loss_dice = self.dice_loss(output_labeled_soft, labeled_label.unsqueeze(1))
            sup_loss = 0.5 * (loss_dice + loss_ce)

        # U2PL official: sup + unsup + contrast (no extra consistency_weight rampup)
        total_loss = sup_loss + unsup_loss + self.contrast_weight * contra_loss

        return {
            'total': total_loss,
            'ce': loss_ce,
            'dice': loss_dice,
            'consistency': unsup_loss + contra_loss,
            'consistency_weight': 1.0
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        self._backward_and_step(loss_dict['total'], optimizer=self.optimizer)
        self._update_ema(iter_num)
        return loss_dict

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        if self.ema_model is not None:
            self.ema_model.eval()
        self.projector.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "ema_model" in state_dict:
                self.ema_model.load_state_dict(state_dict["ema_model"])
            if "projector" in state_dict:
                self.projector.load_state_dict(state_dict["projector"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--u2pl_drop_percent', type=float, default=80)
        parser.add_argument('--u2pl_contrast_weight', type=float, default=0.1)
        parser.add_argument('--u2pl_low_entropy_threshold', type=float, default=20.0)
        parser.add_argument('--u2pl_num_queries', type=int, default=256)
        parser.add_argument('--u2pl_num_negatives', type=int, default=50)
        parser.add_argument('--u2pl_temperature', type=float, default=0.5)
        parser.add_argument('--u2pl_class_threshold', type=float, default=0.3)
        parser.add_argument('--u2pl_class_neg_threshold', type=float, default=1.0)
        parser.add_argument('--u2pl_low_rank', type=int, default=1)
        parser.add_argument('--u2pl_high_rank', type=int, default=2)
        parser.add_argument('--u2pl_sup_only_epoch', type=int, default=0)
