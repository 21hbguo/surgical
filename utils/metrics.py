import numpy as np
import torch
import torch.nn.functional as F


METRIC_PAIRS = (
    ("Dice", "dice"),
    ("IoU", "iou"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Acc", "acc"),
    ("HD95", "hd95"),
    ("ASD", "asd"),
)


def dice_score(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_dice_per_class(pred, target, num_classes):
    dice_scores = []
    if num_classes == 1 or pred.shape[1] == 1:
        if pred.min() >= 0 and pred.max() <= 1:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
        dice_scores.append(dice_score(pred_binary, target).item())
    else:
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        for c in range(1, num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            dice_scores.append(dice_score(pred_c, target_c).item())
    return dice_scores


def resize_mask_to_shape(mask, target_shape):
    mask = np.asarray(mask)
    target_h, target_w = tuple(int(v) for v in target_shape[:2])
    if mask.shape == (target_h, target_w):
        return mask
    import cv2
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def calculate_segmentation_case_metrics(pred, gt, num_classes):
    pred = np.asarray(pred).astype(np.uint8)
    gt = np.asarray(gt).astype(np.uint8)
    if pred.shape != gt.shape:
        pred = resize_mask_to_shape(pred, gt.shape)
    records = []
    for cls in range(1, num_classes):
        pred_cls = pred == cls
        gt_cls = gt == cls
        if pred_cls.sum() == 0 and gt_cls.sum() == 0:
            records.append({
                "Dice": 1.0,
                "IoU": 1.0,
                "TP": 0.0,
                "FP": 0.0,
                "FN": 0.0,
                "Acc": float((pred_cls == gt_cls).mean()),
                "HD95": 0.0,
                "ASD": 0.0,
                "Valid": True,
                "Class": cls,
            })
            continue
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        smooth = 1e-6
        dice = (2.0 * intersection + smooth) / (pred_cls.sum() + gt_cls.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)
        tp = float(intersection)
        fp = float(np.logical_and(pred_cls, np.logical_not(gt_cls)).sum())
        fn = float(np.logical_and(np.logical_not(pred_cls), gt_cls).sum())
        acc = float((pred_cls == gt_cls).mean())
        records.append({
            "Dice": dice,
            "IoU": iou,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Acc": acc,
            "HD95": float("nan"),
            "ASD": float("nan"),
            "Valid": True,
            "Class": cls,
        })
    return records


def _ensure_depth3(tensor):
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    if tensor.shape[1] > 3:
        return tensor[:, :3]
    pad_c = 3 - tensor.shape[1]
    return F.pad(tensor, (0, 0, 0, 0, 0, pad_c), mode="constant", value=0.0)


def _build_ssim_window(channels, device, dtype, window_size=11, sigma=1.5):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim(pred, target, data_range=1.0, window_size=11):
    pred = _ensure_depth3(pred)
    target = _ensure_depth3(target)

    channels = pred.shape[1]
    window = _build_ssim_window(channels, pred.device, pred.dtype, window_size=window_size)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()


def compute_depth_psnr_ssim(pred, target):
    pred = pred.float()
    target = target.float()

    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=False)

    pred = _ensure_depth3(pred)
    target = _ensure_depth3(target)

    B = target.shape[0]
    data_min = target.view(B, -1).min(dim=1).values
    data_max = target.view(B, -1).max(dim=1).values
    data_range = (data_max - data_min).clamp(min=1e-6)

    mse = ((pred - target) ** 2).view(B, -1).mean(dim=1)
    psnr_per_sample = 10.0 * torch.log10((data_range * data_range) / (mse + 1e-8))
    psnr = psnr_per_sample.mean()
    ssim = _ssim(pred, target, data_range=float(data_range.mean().detach().item()))

    return {
        "psnr": float(psnr.detach().item()),
        "ssim": float(ssim.detach().item()),
    }
