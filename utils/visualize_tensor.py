import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from .common import DEFAULT_COLORMAP


@dataclass
class TensorItem:
    tensor: torch.Tensor
    type: str
    name: str = ""


def _build_colormap(num_classes=None):
    total = int(num_classes) if num_classes else len(DEFAULT_COLORMAP)
    return {idx: DEFAULT_COLORMAP[idx % len(DEFAULT_COLORMAP)] for idx in range(total)}


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] <= 4 and tensor.shape[1] > 4 and tensor.shape[2] > 4:
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(1)
    return tensor


def _normalize_01(tensor: torch.Tensor) -> torch.Tensor:
    t_min = tensor.amin((1, 2, 3), keepdim=True)
    t_max = tensor.amax((1, 2, 3), keepdim=True)
    normalized = torch.where(t_max > t_min, (tensor - t_min) / (t_max - t_min + 1e-8), tensor)
    return normalized.clamp(0, 1)


def _labels_to_rgb(labels: torch.Tensor, colormap: dict[int, list[int]]):
    labels = labels.long()
    num_colors = max(colormap.keys()) + 1 if colormap else 1
    colors = torch.tensor(
        [colormap.get(idx, [0, 0, 0]) for idx in range(num_colors)],
        dtype=torch.float32,
        device=labels.device,
    )
    labels = labels.clamp(0, num_colors - 1)
    return colors[labels].permute(0, 3, 1, 2) / 255.0


def _to_label_batch(tensor: torch.Tensor, from_prediction: bool) -> torch.Tensor:
    tensor = tensor.detach().clone()
    if from_prediction:
        tensor = _ensure_4d(tensor)
        if tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return torch.argmax(tensor, dim=1)

    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] <= 4 and tensor.shape[1] > 4 and tensor.shape[2] > 4:
            if tensor.shape[0] == 1:
                return tensor
            return torch.argmax(tensor, dim=0, keepdim=False).unsqueeze(0)
        return tensor
    if tensor.dim() == 4:
        if tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return torch.argmax(tensor, dim=1)
    return tensor


def _process_item_tensor(tensor: torch.Tensor, item_type: str, colormap: dict[int, list[int]]) -> tuple[torch.Tensor, bool]:
    if item_type == "label":
        labels = _to_label_batch(tensor, from_prediction=False)
        return _labels_to_rgb(labels, colormap), True

    if item_type == "prediction":
        labels = _to_label_batch(tensor, from_prediction=True)
        return _labels_to_rgb(labels, colormap), True

    tensor = _ensure_4d(tensor.detach().clone()).float()
    if item_type == "image":
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] >= 4:
            tensor = tensor[:, :3]
        return tensor, False

    if item_type in {"gray", "depth"}:
        if tensor.shape[1] > 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        return _normalize_01(tensor).repeat(1, 3, 1, 1), False

    if item_type == "feature":
        channels = tensor.shape[1]
        if channels < 3:
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, 0, 0, 3 - channels))
        elif channels > 3:
            tensor = tensor[:, :3]
        return _normalize_01(tensor), False

    return tensor, False


def visualize_tensors(items, save_path, nrow=4, filename="viz.png", colormap=None):
    if not items:
        return
    os.makedirs(save_path, exist_ok=True)
    colormap = colormap or _build_colormap()

    processed_tensors = []
    is_discrete = []
    for item in items:
        tensor, discrete = _process_item_tensor(item.tensor, item.type, colormap)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        processed_tensors.append(tensor)
        is_discrete.append(discrete)
    if not processed_tensors:
        return

    ref_size = processed_tensors[0].shape[2:]
    resized_tensors = []
    for tensor, discrete in zip(processed_tensors, is_discrete):
        if tensor.shape[2:] == ref_size:
            resized_tensors.append(tensor)
            continue
        if discrete:
            resized_tensors.append(F.interpolate(tensor, size=ref_size, mode="nearest"))
        else:
            resized_tensors.append(F.interpolate(tensor, size=ref_size, mode="bilinear", align_corners=False))

    min_batch = min(tensor.shape[0] for tensor in resized_tensors)
    if min_batch <= 0:
        return
    resized_tensors = [tensor[:min_batch] for tensor in resized_tensors]
    grid = make_grid(torch.cat(resized_tensors, dim=0), nrow=nrow, padding=5)
    save_image(grid, os.path.join(save_path, filename))


def visualize_input_columns(column_tensors, save_path, filename="latest_inputs.png", max_rows=4):
    if not column_tensors:
        return

    expanded_columns = []
    for name, tensor in column_tensors:
        if tensor is None:
            continue
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4 or tensor.shape[0] == 0:
            continue
        channels = tensor.shape[1]
        if channels <= 3:
            expanded_columns.append((name, tensor))
        elif channels == 4:
            expanded_columns.append((f"{name}_rgb", tensor[:, :3]))
            expanded_columns.append((f"{name}_depth", tensor[:, 3:4]))
        else:
            expanded_columns.append((f"{name}_rgb", tensor[:, :3]))
            expanded_columns.append((f"{name}_depth", tensor[:, 3:]))
    if not expanded_columns:
        return

    total_rows = max(tensor.shape[0] for _, tensor in expanded_columns)
    rows = total_rows if (max_rows is None or max_rows <= 0) else min(max_rows, total_rows)
    if rows <= 0:
        return

    padded_columns = []
    for name, tensor in expanded_columns:
        if tensor.shape[0] < rows:
            pad = tensor.new_zeros((rows - tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            tensor = torch.cat([tensor, pad], dim=0)
        padded_columns.append((name, tensor))

    items = []
    for row_idx in range(rows):
        for _, tensor in padded_columns:
            items.append(TensorItem(tensor[row_idx].detach().cpu(), type="image"))
    visualize_tensors(items, save_path=save_path, nrow=len(padded_columns), filename=filename)

    os.makedirs(save_path, exist_ok=True)
    sidecar = os.path.join(save_path, f"{filename}.columns.txt")
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write("\n".join(name for name, _ in padded_columns) + "\n")
