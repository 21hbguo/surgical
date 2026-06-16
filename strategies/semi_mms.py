import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_strategy import BaseTrainingStrategy


class Projector(nn.Module):
    """MMS projector for contrastive learning."""
    def __init__(self, in_channels=2, mid_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PixelContrastiveLoss(nn.Module):
    """Pixel-wise contrastive loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        if feat_q.shape[2] > 32 or feat_q.shape[3] > 32:
            feat_q = F.adaptive_avg_pool2d(feat_q, 32)
            feat_k = F.adaptive_avg_pool2d(feat_k, 32)
        B, C, H, W = feat_q.shape
        feat_q = feat_q.view(B, C, -1).permute(0, 2, 1)
        feat_k = feat_k.view(B, C, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=2)
        feat_k = F.normalize(feat_k.detach(), dim=-1, p=2)

        l_pos = torch.bmm(feat_q.reshape(-1, 1, C), feat_k.reshape(-1, C, 1)).view(-1, 1)
        feat_q_r = feat_q.reshape(B, -1, C)
        feat_k_r = feat_k.reshape(B, -1, C)
        l_neg = torch.bmm(feat_q_r, feat_k_r.transpose(2, 1))
        npatches = l_neg.size(1)
        diagonal = torch.eye(npatches, device=l_neg.device, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(diagonal, -10.0)
        l_neg = l_neg.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        loss = self.ce(out, torch.zeros(out.size(0), dtype=torch.long, device=out.device))
        return loss


class MMSStrategy(BaseTrainingStrategy):
    """Min-Max Similarity (TMI 2023).

    Faithful reproduction:
    - Dual independent models (model_1 + model_2)
    - Pixel contrastive loss between projected features from different models
    - Soft consistency difference loss on unlabeled data
    - Edge-aware weighted CE + Dice supervised loss
    """

    def __init__(self, args, model, optimizer, device, scaler=None):
        super().__init__(args, model, optimizer, device, scaler=scaler)

        # Create second independent model
        self.model2 = self._create_independent_model()

        # Projector heads (input = num_classes logits)
        feat_ch = args.num_classes
        self.projector_1 = Projector(feat_ch, 64, 128).to(device)
        self.projector_2 = Projector(feat_ch, 64, 128).to(device)

        self.contrast_loss = PixelContrastiveLoss().to(device)

        # Single optimizer for all parameters
        all_params = (
            list(self.model.parameters()) +
            list(self.model2.parameters()) +
            list(self.projector_1.parameters()) +
            list(self.projector_2.parameters())
        )
        self.optimizer = torch.optim.Adam(
            all_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001
        )

        self.contrast_weight = args.mms_contrast_weight
        self.diff_weight = args.mms_diff_weight

    def _create_independent_model(self):
        """Create a fresh model with random weights."""
        orig_model = getattr(self.model, "_orig_mod", self.model)
        sig = inspect.signature(type(orig_model).__init__)
        init_kwargs = {}
        model_params = getattr(orig_model, "params", None)

        if model_params is not None:
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                value = self._resolve_model_param_value(model_params, param_name, param)
                if value != inspect.Parameter.empty:
                    init_kwargs[param_name] = value

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name not in init_kwargs and hasattr(orig_model, param_name):
                init_kwargs[param_name] = getattr(orig_model, param_name)

        new_model = type(orig_model)(**init_kwargs)
        if not hasattr(new_model, "params") and hasattr(self.model, "params"):
            new_model.params = self.model.params
        return new_model.to(self.device)

    def _weighted_ce_loss(self, pred, mask):
        """Edge-aware weighted CE loss for multi-class segmentation.

        Uses one-hot encoded mask to detect edges across all classes,
        avoiding the issue of treating class indices as continuous values.
        """
        num_classes = pred.shape[1]
        mask_oh = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
        # Edge detection: avg pool smooths, difference detects boundaries
        edge = torch.abs(
            F.avg_pool2d(mask_oh, kernel_size=31, stride=1, padding=15) - mask_oh
        )
        # Max across classes gives unified edge map
        edge_map = edge.max(dim=1, keepdim=True)[0]
        weit = 1 + 5 * edge_map
        ce = F.cross_entropy(pred, mask.long(), reduction='none')
        return (weit.squeeze(1) * ce).sum() / weit.sum()

    def _split_labeled(self, labeled_data, labeled_label):
        """Split labeled batch into two non-overlapping halves."""
        B = labeled_data.shape[0]
        half = B // 2
        return (labeled_data[:half], labeled_label[:half],
                labeled_data[half:], labeled_label[half:])

    def compute_loss(self, batch_data, iter_num=0, epoch=0):
        volume = batch_data["image"].to(self.device)
        depth_tensor = self._get_depth_tensor(batch_data)
        if depth_tensor is not None:
            volume = torch.cat([volume, depth_tensor], dim=1)
        label = batch_data["label"].to(self.device)

        labeled_data = volume[:self.labeled_bs]
        labeled_label = label[:self.labeled_bs]
        unlabeled_data = volume[self.labeled_bs:]

        # Split labeled data into two non-overlapping subsets
        inputs_S1, labels_S1, inputs_S2, labels_S2 = self._split_labeled(
            labeled_data, labeled_label
        )

        if inputs_S1.shape[0] == 0:
            return {'total': torch.tensor(0.0, device=self.device),
                    'ce': torch.tensor(0.0), 'dice': torch.tensor(0.0),
                    'consistency': torch.tensor(0.0), 'consistency_weight': 0.0}

        # Model 1 on split 1, Model 2 on split 2
        pred_s1 = self.model(inputs_S1)
        if isinstance(pred_s1, tuple):
            pred_s1 = pred_s1[0]

        pred_s2 = self.model2(inputs_S2)
        if isinstance(pred_s2, tuple):
            pred_s2 = pred_s2[0]

        # Supervised loss (edge-aware weighted CE + Dice)
        dice_s1 = self.dice_loss(torch.softmax(pred_s1, dim=1), labels_S1.unsqueeze(1))
        dice_s2 = self.dice_loss(torch.softmax(pred_s2, dim=1), labels_S2.unsqueeze(1))
        loss_sup = (self._weighted_ce_loss(pred_s1, labels_S1) + dice_s1 +
                    self._weighted_ce_loss(pred_s2, labels_S2) + dice_s2)

        # Unlabeled predictions from both models
        consistency_weight = self._get_consistency_weight(iter_num)
        loss_diff = torch.tensor(0.0, device=self.device)
        loss_contrast = torch.tensor(0.0, device=self.device)

        if unlabeled_data.shape[0] > 0:
            u_pred1 = self.model(unlabeled_data)
            if isinstance(u_pred1, tuple):
                u_pred1 = u_pred1[0]
            u_pred1_soft = torch.softmax(u_pred1, dim=1)

            u_pred2 = self.model2(unlabeled_data)
            if isinstance(u_pred2, tuple):
                u_pred2 = u_pred2[0]
            u_pred2_soft = torch.softmax(u_pred2, dim=1)

            # Soft consistency difference loss (KL divergence between models)
            # This preserves gradient flow unlike hard argmax
            loss_diff = 0.5 * (
                F.kl_div(u_pred1_soft.log(), u_pred2_soft.detach(), reduction='mean') +
                F.kl_div(u_pred2_soft.log(), u_pred1_soft.detach(), reduction='mean')
            )

            # Pixel contrastive loss (features from DIFFERENT models)
            feat_q = self.projector_1(u_pred1)
            feat_k = self.projector_2(u_pred2.detach())
            loss_contrast = self.contrast_loss(feat_q, feat_k)

        total_loss = (0.25 * loss_sup +
                      self.diff_weight * consistency_weight * loss_diff +
                      self.contrast_weight * consistency_weight * loss_contrast)

        ce = self.ce_loss(pred_s1, labels_S1[:pred_s1.shape[0]].long())
        dice = self.dice_loss(torch.softmax(pred_s1, dim=1),
                              labels_S1[:pred_s1.shape[0]].unsqueeze(1))

        return {
            'total': total_loss,
            'ce': ce,
            'dice': dice,
            'consistency': loss_diff + loss_contrast,
            'consistency_weight': consistency_weight
        }

    def training_step(self, batch_data, iter_num, epoch=0):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_dict = self.compute_loss(batch_data, iter_num, epoch)
        loss = loss_dict['total']
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._all_params(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._all_params(), self.grad_clip)
            self.optimizer.step()
        return loss_dict

    def _all_params(self):
        return (
            list(self.model.parameters()) +
            list(self.model2.parameters()) +
            list(self.projector_1.parameters()) +
            list(self.projector_2.parameters())
        )

    def _set_model_mode(self, training):
        self.model.train(mode=training)
        self.model2.train(mode=training)

    def get_state_dict(self):
        return {
            "model": self.model.state_dict(),
            "model2": self.model2.state_dict(),
            "projector_1": self.projector_1.state_dict(),
            "projector_2": self.projector_2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
            if "model2" in state_dict:
                self.model2.load_state_dict(state_dict["model2"])
            if "projector_1" in state_dict:
                self.projector_1.load_state_dict(state_dict["projector_1"])
            if "projector_2" in state_dict:
                self.projector_2.load_state_dict(state_dict["projector_2"])
            return
        self.model.load_state_dict(state_dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--mms_contrast_weight', type=float, default=0.25)
        parser.add_argument('--mms_diff_weight', type=float, default=0.25)
