
# 脚本工具：执行 感受野可视化 相关的数据或分析任务。

import argparse
import importlib
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


# 构建模型。
def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet34":
        model = tv_models.resnet34(weights=None)
        model.load_state_dict(torch.load(RESNET34_WEIGHTS, map_location="cpu"), strict=True)
        return model.eval()
    if model_name == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.load_state_dict(torch.load(RESNET50_WEIGHTS, map_location="cpu"), strict=True)
        return model.eval()
    if model_name == "dinov2_vits14":
        return dinov2_vits14(pretrained=True, weights=DINOV2_WEIGHTS).eval()
    if model_name == "dinov3_vits16":
        return dinov3_vits16(pretrained=True, weights=DINOV3_WEIGHTS).eval()
    model = getattr(tv_models, model_name)(weights=None)
    return model.eval()


# 获取相关内容。
def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    named = dict(model.named_modules())
    return named[layer_name]


# 加载图像。
def load_image(image_path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    tensor = T.ToTensor()(image).unsqueeze(0)
    return tensor, image_np


# 计算相关内容。
def _compute_vit_rf_heatmap(model: nn.Module, model_name: str, image_tensor: torch.Tensor, unit_x: int, unit_y: int):
    image_tensor = image_tensor.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    features = model.forward_features(image_tensor)
    patch_tokens = features["x_norm_patchtokens"]
    num_patches = patch_tokens.shape[1]
    side = int(round(num_patches ** 0.5))
    feat = patch_tokens.transpose(1, 2).reshape(1, patch_tokens.shape[-1], side, side)
    h, w = feat.shape[-2:]
    unit_y = h // 2 if unit_y < 0 else max(0, min(unit_y, h - 1))
    unit_x = w // 2 if unit_x < 0 else max(0, min(unit_x, w - 1))
    score = feat[0, :, unit_y, unit_x].sum()
    score.backward()
    grad = image_tensor.grad.detach().abs()[0]
    heatmap = grad.max(dim=0).values.cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap, (unit_x, unit_y), (w, h)


# 计算相关内容。
def compute_rf_heatmap(model: nn.Module, layer_name: str, image_tensor: torch.Tensor, unit_x: int, unit_y: int):
    activations = {}

    # 处理相关内容相关逻辑。
    def hook(_, __, output):
        activations["feat"] = output

    handle = get_layer(model, layer_name).register_forward_hook(hook)
    image_tensor = image_tensor.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    _ = model(image_tensor)
    handle.remove()

    feat = activations["feat"]

    h, w = feat.shape[-2:]
    unit_y = h // 2 if unit_y < 0 else max(0, min(unit_y, h - 1))
    unit_x = w // 2 if unit_x < 0 else max(0, min(unit_x, w - 1))

    score = feat[0, :, unit_y, unit_x].sum()
    score.backward()
    grad = image_tensor.grad.detach().abs()[0]
    heatmap = grad.max(dim=0).values.cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap, (unit_x, unit_y), (w, h)


# 保存相关内容。
def save_visualization(image_np: np.ndarray, heatmap: np.ndarray, save_path: str, model_name: str, layer_name: str, unit_xy, feat_hw):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title("Effective RF Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{model_name} | {layer_name}", fontsize=11, y=0.98)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# 解析训练脚本的命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize effective receptive field by backpropagating a single unit.")
    parser.add_argument("--image-path", required=True, help="Path to an RGB image.")
    parser.add_argument(
        "--model-name",
        default="resnet34",
        help="Model name: resnet34, resnet50, dinov2_vits14, dinov3_vits16.",
    )
    parser.add_argument(
        "--layer-name",
        default="layer4",
        help="Target feature layer name for CNNs, e.g. layer4, layer3, features.28. Ignored for DINO models.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize input image to a square size.")
    parser.add_argument("--unit-x", type=int, default=-1, help="Feature x index. -1 means center.")
    parser.add_argument("--unit-y", type=int, default=-1, help="Feature y index. -1 means center.")
    parser.add_argument("--save-path", required=True, help="Output figure path.")
    return parser.parse_args()
# 组织脚本主流程。


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(args.model_name).to(device)
    image_tensor, image_np = load_image(args.image_path, args.image_size)
    image_tensor = image_tensor.to(device)
    if args.model_name in {"dinov2_vits14", "dinov3_vits16"}:
        heatmap, unit_xy, feat_hw = _compute_vit_rf_heatmap(
            model=model,
            model_name=args.model_name,
            image_tensor=image_tensor,
            unit_x=args.unit_x,
            unit_y=args.unit_y,
        )
        title_layer_name = "x_norm_patchtokens"
    else:
        heatmap, unit_xy, feat_hw = compute_rf_heatmap(
            model=model,
            layer_name=args.layer_name,
            image_tensor=image_tensor,
            unit_x=args.unit_x,
            unit_y=args.unit_y,
        )
        title_layer_name = args.layer_name
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    save_visualization(image_np, heatmap, args.save_path, args.model_name, title_layer_name, unit_xy, feat_hw)
    print(f"Visualization saved to: {args.save_path}")


if __name__ == "__main__":
    main()
