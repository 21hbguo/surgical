
# 脚本工具：执行 注意力可视化 相关的数据或分析任务。

import argparse
import importlib
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

DINOV2_REPO = "/home/guo/project/other_method/SSL/dinov2"
DINOV3_REPO = "/home/guo/project/other_method/SSL/dinov3"
RESNET34_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/resnet34-b627a593.pth"
RESNET50_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/resnet50-11ad3fa6.pth"
DINOV2_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/dinov2_small.pth"
DINOV3_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)

dinov2_vits14 = importlib.import_module("dinov2.hub.backbones").dinov2_vits14
dinov3_vits16 = importlib.import_module("dinov3.hub.backbones").dinov3_vits16


# 解析训练脚本的命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention for ResNet and DINO backbones.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["resnet34", "resnet50", "dinov2_vits14", "dinov3_vits16"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--layout", choices=["horizontal", "vertical", "grid"], default="vertical")
    return parser.parse_args()


# 构建模型。
def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet34":
        model = tv_models.resnet34(weights=None)
        state_dict = torch.load(RESNET34_WEIGHTS, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model.eval()
    if model_name == "resnet50":
        model = tv_models.resnet50(weights=None)
        state_dict = torch.load(RESNET50_WEIGHTS, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model.eval()
    if model_name == "dinov2_vits14":
        return dinov2_vits14(pretrained=True, weights=DINOV2_WEIGHTS).eval()
    if model_name == "dinov3_vits16":
        return dinov3_vits16(pretrained=True, weights=DINOV3_WEIGHTS).eval()
    return getattr(tv_models, model_name)(weights=None).eval()


# 加载图像。
def load_image(image_path: str, image_size: int, device: torch.device):
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    return tensor, image_np


# 处理归一化映射相关逻辑。
def normalize_map(attn: torch.Tensor) -> np.ndarray:
    arr = attn.detach().float().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


# 计算相关内容。
def compute_resnet_gradcam(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    target_layer = model.layer4[-1]
    activations = {}
    gradients = {}

    # 处理相关内容相关逻辑。
    def fwd_hook(_, __, output):
        activations["value"] = output

    # 处理相关内容相关逻辑。
    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    score = logits.max(dim=1).values.sum()
    score.backward()
    h1.remove()
    h2.remove()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True).relu()
    cam = torch.nn.functional.interpolate(
        cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_map(cam)


# 计算dinov2。
def compute_dinov2_attention(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    x = model.prepare_tokens_with_masks(image_tensor, None)
    last_block = model.blocks[-1]
    x_norm = last_block.norm1(x)
    attn_mod = last_block.attn
    B, N, C = x_norm.shape
    qkv = attn_mod.qkv(x_norm).reshape(B, N, 3, attn_mod.num_heads, C // attn_mod.num_heads)
    q, k, _ = torch.unbind(qkv, 2)
    q, k = [t.transpose(1, 2) for t in [q, k]]
    attn = (q @ k.transpose(-2, -1)) * attn_mod.scale
    attn = attn.softmax(dim=-1)
    cls_attn = attn[0, :, 0, 1 + model.num_register_tokens :].mean(dim=0)
    side = int(math.sqrt(cls_attn.numel()))
    attn_map = cls_attn.reshape(side, side).unsqueeze(0).unsqueeze(0)
    attn_map = torch.nn.functional.interpolate(
        attn_map, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_map(attn_map)


# 计算dinov3。
def compute_dinov3_attention(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    x, (H, W) = model.prepare_tokens_with_masks(image_tensor, None)
    last_block = model.blocks[-1]
    rope = model.rope_embed(H=H, W=W) if model.rope_embed is not None else None
    x_norm = last_block.norm1(x)
    attn_mod = last_block.attn
    B, N, C = x_norm.shape
    qkv = attn_mod.qkv(x_norm).reshape(B, N, 3, attn_mod.num_heads, C // attn_mod.num_heads)
    q, k, _ = torch.unbind(qkv, 2)
    q, k = [t.transpose(1, 2) for t in [q, k]]
    if rope is not None:
        q, k = attn_mod.apply_rope(q, k, rope)
    attn = (q @ k.transpose(-2, -1)) * attn_mod.scale
    attn = attn.softmax(dim=-1)
    cls_attn = attn[0, :, 0, 1 + model.n_storage_tokens :].mean(dim=0)
    side = int(math.sqrt(cls_attn.numel()))
    attn_map = cls_attn.reshape(side, side).unsqueeze(0).unsqueeze(0)
    attn_map = torch.nn.functional.interpolate(
        attn_map, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_map(attn_map)


# 计算映射。
def compute_attention_map(model_name: str, model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    if model_name in {"resnet34", "resnet50"}:
        return compute_resnet_gradcam(model, image_tensor)
    if model_name == "dinov2_vits14":
        return compute_dinov2_attention(model, image_tensor)
    if model_name == "dinov3_vits16":
        return compute_dinov3_attention(model, image_tensor)
    return compute_resnet_gradcam(model, image_tensor)


# 处理相关内容相关逻辑。
def render_panel(image_np: np.ndarray, attn_map: np.ndarray, title: str) -> np.ndarray:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(image_np)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(attn_map, cmap="hot")
    axes[1].set_title("Attention", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(attn_map, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11, y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    panel = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return panel


# 处理相关内容相关逻辑。
def stitch_panels(panels: list[np.ndarray], titles: list[str], save_path: str, layout: str):
    if layout == "grid":
        rows, cols = 2, math.ceil(len(panels) / 2)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows))
        axes = np.array(axes).reshape(rows, cols)
        flat_axes = axes.flatten()
        for ax, panel, title in zip(flat_axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
        for ax in flat_axes[len(panels):]:
            ax.axis("off")
    elif layout == "horizontal":
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
        axes = [axes] if len(panels) == 1 else axes
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")
    else:
        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.6 * len(panels)))
        axes = [axes] if len(panels) == 1 else axes
        for ax, panel, title in zip(axes, panels, titles):
            ax.imshow(panel)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.03, wspace=0.03)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
# 组织脚本主流程。


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    panels = []
    titles = []
    for model_name in args.model_names:
        print(f"Running {model_name} ...")
        model = build_model(model_name).to(device)
        image_tensor, image_np = load_image(args.image_path, args.image_size, device)
        with torch.set_grad_enabled(True):
            attn_map = compute_attention_map(model_name, model, image_tensor)
        panels.append(render_panel(image_np, attn_map, model_name))
        titles.append(model_name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stitch_panels(panels, titles, args.save_path, args.layout)
    print(f"Visualization saved to: {args.save_path}")


if __name__ == "__main__":
    main()
