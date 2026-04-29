import os

import cv2
import numpy as np
import torch.nn as nn

from utils.common import build_run_output_dir

TEST_RGB_COLORMAP = (
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 128, 128),
    (128, 0, 255),
)


class GradCAMHookManager:
    def __init__(self, model, layer_filter="", all_layers=True, max_layers=0):
        self.model = model
        self.layer_filter = str(layer_filter or "").strip()
        self.all_layers = bool(all_layers)
        self.max_layers = int(max_layers or 0)
        self.handles = []
        self.tensor_handles = {}
        self.layer_names = []
        self.activations = {}
        self.gradients = {}
        self._register()

    def _register(self):
        candidates = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if self.layer_filter and self.layer_filter not in name:
                continue
            candidates.append((name, module))
        if not candidates:
            return
        if not self.all_layers:
            candidates = [candidates[-1]]
        if self.max_layers > 0:
            candidates = candidates[-self.max_layers :]
        for name, module in candidates:
            self.layer_names.append(name)
            handle = module.register_forward_hook(self._make_forward_hook(name))
            self.handles.append(handle)

    def _make_forward_hook(self, name):
        def _hook(_module, _inputs, output):
            self.activations[name] = output.detach().clone()
            if output.requires_grad:
                grad_handle = output.register_hook(self._make_tensor_backward_hook(name))
                self.tensor_handles[name] = grad_handle

        return _hook

    def _make_tensor_backward_hook(self, name):
        def _hook(grad):
            self.gradients[name] = grad

        return _hook

    def clear_cache(self):
        self.activations.clear()
        self.gradients.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for handle in self.tensor_handles.values():
            handle.remove()
        self.handles = []
        self.tensor_handles = {}
        self.layer_names = []
        self.clear_cache()


def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if image.min() < 0.0 or image.max() > 1.0:
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                image = (image - image_min) / (image_max - image_min)
            else:
                image = np.zeros_like(image)
        image = np.clip(image, 0.0, 1.0)
        return (image * 255.0).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def colorize_test_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        color = TEST_RGB_COLORMAP[class_idx % len(TEST_RGB_COLORMAP)]
        rgb[mask == class_idx] = color
    return rgb


def _build_overlay(image: np.ndarray, mask: np.ndarray, num_classes: int, alpha: float = 0.45) -> np.ndarray:
    rgb_image = _prepare_rgb_image(image)
    mask_rgb = colorize_test_mask(mask, num_classes)
    overlay = rgb_image.astype(np.float32).copy()
    mask_pixels = np.asarray(mask) > 0
    overlay[mask_pixels] = (1.0 - alpha) * overlay[mask_pixels] + alpha * mask_rgb[mask_pixels].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_test_rgb_visualization(image, label, pred, output_dir, case_name, mode, num_classes, alpha: float = 0.45):
    pred_overlay = _build_overlay(image, pred, num_classes, alpha=alpha)
    if mode == 2:
        label_overlay = _build_overlay(image, label, num_classes, alpha=alpha)
        result = np.concatenate([label_overlay, pred_overlay], axis=1)
    else:
        result = pred_overlay
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{case_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return save_path


def _build_heatmap_color_image(heat: np.ndarray, target_shape) -> np.ndarray:
    target_h, target_w = tuple(int(v) for v in target_shape[:2])
    if heat is None:
        heat = np.zeros((target_h, target_w), dtype=np.float32)
    heat = np.asarray(heat, dtype=np.float32)
    if heat.shape != (target_h, target_w):
        heat = cv2.resize(heat, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)


def save_multiclass_gradcam_visualization(image, class_heats, output_dir: str, case_name: str, alpha: float = 0.45):
    rgb_image = _prepare_rgb_image(image)
    overlays = [rgb_image]
    heatmaps = [rgb_image]

    for heat in class_heats:
        heat_color = _build_heatmap_color_image(heat, rgb_image.shape[:2])
        overlay = np.clip(
            (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * heat_color.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        overlays.append(overlay)
        heatmaps.append(heat_color)

    result = np.concatenate(
        [
            np.concatenate(overlays, axis=1),
            np.concatenate(heatmaps, axis=1),
        ],
        axis=0,
    )
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{case_name}.png")
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return save_path


def prepare_visual_output_dirs(args, fold, logger):
    rgb_output_dir = None
    feat_output_dir = None
    output_root = None

    if args.rgb or args.feat_vis:
        output_root = build_run_output_dir(args, mode="test", fold=fold)

    if args.rgb:
        rgb_output_dir = os.path.join(output_root, "rgb")
        os.makedirs(rgb_output_dir, exist_ok=True)

    if args.feat_vis:
        feat_output_dir = os.path.join(output_root, "feature_gradcam")
        os.makedirs(feat_output_dir, exist_ok=True)
        logger.info(
            "Feature visualization enabled: dir=%s, method=%s, all_layers=%s, max_layers=%s, max_cases=%s",
            feat_output_dir,
            args.feat_vis_method,
            args.feat_vis_all_layers,
            args.feat_vis_max_layers,
            args.feat_vis_max_cases,
        )

    return rgb_output_dir, feat_output_dir
