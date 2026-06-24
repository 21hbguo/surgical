"""GradCAM visualization for DepthGAN discriminator.

Usage:
    python tools/gradcam_discriminator.py \
        --checkpoint /path/to/model_best.pth \
        --data_dir /path/to/dataset \
        --output_dir /home/guo/project/ssl4mis/data/gradcam \
        --exp endovis2018ISINet/Fully \
        --way fully_supervised_depthGAN \
        --use_depth 1 \
        --num_samples 20 \
        --samples_per_image 10
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.args import finalize_test_args
from core.runtime import resolve_device
from models.factory import create_model
from strategies.fully_supervised_depthGAN import FullySupervisedDepthGANStrategy, DepthGANDiscriminator
from data.dataset import BaseDataSets, H5DataSets


class GradCAM:
    """GradCAM hook for a specific layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self._hook_handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self._hook_handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        # For discriminator, output is (B, 1, H, W), use mean as scalar loss
        loss = output.mean()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def denormalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize image from ImageNet normalization."""
    img = img.copy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return np.clip(img, 0, 1)


def overlay_cam_on_image(image, cam, alpha=0.5):
    """Overlay CAM heatmap on image."""
    cam_resized = np.array(plt.cm.jet(cam)[:, :, :3])
    image = image.transpose(1, 2, 0) if image.ndim == 3 and image.shape[0] == 3 else image
    overlay = (1 - alpha) * image + alpha * cam_resized
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser("Discriminator GradCAM Visualization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="/home/guo/project/ssl4mis/data/gradcam")
    parser.add_argument("--num_samples", type=int, default=20, help="Total samples to visualize")
    parser.add_argument("--samples_per_image", type=int, default=10, help="Rows per output image")
    parser.add_argument("--target_layer_idx", type=int, default=8, help="Discriminator conv layer index for GradCAM")
    parser.add_argument("--split", default="val", help="Dataset split to use")
    parser.add_argument("--fold", type=int, default=0, help="Dataset fold index")
    parser.add_argument("--device", default="cuda")
    # Model/data args (will be overridden from checkpoint)
    parser.add_argument("--exp", default=None)
    parser.add_argument("--way", default="fully_supervised_depthGAN")
    parser.add_argument("--model", default=None)
    parser.add_argument("--pretrain", default="imagenet")
    parser.add_argument("--use_depth", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--task", default="organ")
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--depth_uint", type=int, default=16)
    parser.add_argument("--normalize", default="imagenet")
    parser.add_argument("--resize_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--gan_loss_weight", type=float, default=0.1)
    parser.add_argument("--gan_lr", type=float, default=1e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # Merge saved args with CLI args (CLI overrides)
    for key in ["exp", "way", "model", "pretrain", "use_depth", "num_classes",
                "task", "root_path", "depth_uint", "normalize", "gan_loss_weight", "gan_lr"]:
        if getattr(args, key, None) is None and key in saved_args:
            setattr(args, key, saved_args[key])

    if args.exp is None:
        raise ValueError("--exp is required (or must be in checkpoint)")
    # root_path from checkpoint is correct, don't override it
    if args.root_path is None:
        from core.args import infer_root_path_from_exp
        args.root_path = infer_root_path_from_exp(args.exp)
    # Fix exp to match actual data directory name (remove normalize suffix)
    args.exp = args.exp.replace("_255/", "/").replace("_255", "")

    # Set required args for finalize_test_args
    args.result_root = args.output_dir
    args.train_result_root = args.output_dir
    args.snapshot_path = os.path.dirname(args.checkpoint)
    args.requested_checkpoint_type = "best"
    args.no_val = False
    # fold is already set from argparse
    args.use_val = True
    args.sampling = "none"
    args.for_inference = False
    args.seed = 42
    # Additional attrs needed by _finalize_common_args and create_model
    args.pretrain_root = ""
    args.depth_pretrain_path = ""
    args.dformerv2_pretrain_path = ""
    args.dinov3_repo_dir = ""
    args.freeze = False
    args.compile = False
    if not hasattr(args, "filter_num"):
        args.filter_num = 16  # default UNet base filter num
    if not hasattr(args, "resnet_variant"):
        args.resnet_variant = "resnet50"
    if not hasattr(args, "dinov3_weights"):
        args.dinov3_weights = ""
    if not hasattr(args, "consistency"):
        args.consistency = 0.1
    if not hasattr(args, "consistency_rampup"):
        args.consistency_rampup = 40.0
    if not hasattr(args, "pseudo_threshold"):
        args.pseudo_threshold = 0.95
    if not hasattr(args, "num_heads"):
        args.num_heads = 4
    if not hasattr(args, "feature_dim"):
        args.feature_dim = 128
    if not hasattr(args, "proto_temperature"):
        args.proto_temperature = 0.1
    if not hasattr(args, "contrast_feature_dim"):
        args.contrast_feature_dim = 128
    if not hasattr(args, "proto_feature_dim"):
        args.proto_feature_dim = 128
    if not hasattr(args, "grad_clip"):
        args.grad_clip = 12.0
    if not hasattr(args, "strong"):
        args.strong = 0
    if not hasattr(args, "labeled_bs"):
        args.labeled_bs = 2
    if not hasattr(args, "unlabeled_bs"):
        args.unlabeled_bs = 4
    if not hasattr(args, "ema_decay"):
        args.ema_decay = 0.99
    if not hasattr(args, "lr"):
        args.lr = 1e-4
    if not hasattr(args, "lr_scheduler"):
        args.lr_scheduler = "poly"
    if not hasattr(args, "lr_warmup_iters"):
        args.lr_warmup_iters = 0
    if not hasattr(args, "amp"):
        args.amp = False
    if not hasattr(args, "max_iterations"):
        args.max_iterations = 1000
    if not hasattr(args, "val_iter"):
        args.val_iter = 100
    if not hasattr(args, "early_stopping"):
        args.early_stopping = 0.0

    args = finalize_test_args(args)

    # Create model and strategy
    model = create_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    strategy = FullySupervisedDepthGANStrategy(args, model, optimizer, device)

    # Load weights
    model_state = ckpt.get("model_state", ckpt)
    if isinstance(model_state, dict) and "model" in model_state:
        strategy.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    strategy.eval()

    # Create dataset
    dataset = H5DataSets(
        base_dir=args.root_path,
        split=args.split,
        transform=None,
        fold=args.fold,
        depth_channels=args.use_depth if args.use_depth else None,
        depth_uint=args.depth_uint,
        use_val=True,
        task=args.task,
    )

    num_samples = min(args.num_samples, len(dataset))
    samples_per_image = args.samples_per_image
    num_images = (num_samples + samples_per_image - 1) // samples_per_image

    print(f"Generating {num_images} images with {num_samples} total samples...")

    # Set up GradCAM on discriminator
    disc = strategy.discriminator
    target_layer = disc.net[args.target_layer_idx]
    print(f"Target layer for GradCAM: {target_layer}")

    for img_idx in range(num_images):
        start = img_idx * samples_per_image
        end = min(start + samples_per_image, num_samples)
        n_rows = end - start

        fig = plt.figure(figsize=(20, 3.5 * n_rows))
        gs = GridSpec(n_rows, 5, figure=fig, hspace=0.3, wspace=0.15)

        for row, sample_idx in enumerate(range(start, end)):
            sample = dataset[sample_idx]
            image = sample["image"]  # (C, H, W) normalized
            label = sample["label"]  # (H, W)
            depth_key = "depth1" if args.use_depth == 1 else "depth3"
            depth = sample.get(depth_key)

            # Convert to tensors if needed
            if not isinstance(image, torch.Tensor):
                image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            else:
                image_tensor = image.unsqueeze(0).to(device)

            if depth is not None and not isinstance(depth, torch.Tensor):
                depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)
            elif depth is not None:
                depth_tensor = depth.unsqueeze(0).to(device)
            else:
                depth_tensor = None

            with torch.no_grad():
                output = strategy.model(image_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                output_soft = F.softmax(output, dim=1)
                pred_mask = output_soft.argmax(dim=1).squeeze(0).cpu().numpy()

            # Build both real and fake discriminator inputs
            if isinstance(label, torch.Tensor):
                label_np = label.cpu().numpy()
            else:
                label_np = label

            if depth_tensor is not None:
                label_onehot = F.one_hot(torch.from_numpy(label_np).long(), num_classes=args.num_classes)  # (H, W, C)
                label_onehot = label_onehot.permute(2, 0, 1).unsqueeze(0).to(image_tensor.dtype).to(device)  # (1, C, H, W)
                if depth_tensor.shape[1] == 1:
                    real_depth = label_onehot * depth_tensor
                    fake_depth = output_soft * depth_tensor
                else:
                    real_depth = (label_onehot.unsqueeze(2) * depth_tensor.unsqueeze(1)).flatten(1, 2)
                    fake_depth = (output_soft.unsqueeze(2) * depth_tensor.unsqueeze(1)).flatten(1, 2)
            else:
                real_depth = label_onehot
                fake_depth = output_soft

            # Compute GradCAM and discriminator logits for both real and fake
            from PIL import Image
            H, W = label_np.shape

            # Get discriminator logits
            with torch.no_grad():
                logit_fake = disc(fake_depth).mean().item()
                logit_real = disc(real_depth).mean().item()

            gradcam = GradCAM(disc, target_layer)

            cam_fake = gradcam.compute(fake_depth, target_class=0)
            cam_fake_pil = Image.fromarray(cam_fake)
            cam_fake_resized = np.array(cam_fake_pil.resize((W, H), Image.BILINEAR))

            cam_real = gradcam.compute(real_depth, target_class=0)
            cam_real_pil = Image.fromarray(cam_real)
            cam_real_resized = np.array(cam_real_pil.resize((W, H), Image.BILINEAR))

            gradcam.remove_hooks()

            # Prepare images for display
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = image
            image_show = denormalize_image(image_np)
            image_show = image_show.transpose(1, 2, 0)  # (H, W, C)

            if depth is not None:
                if isinstance(depth, torch.Tensor):
                    depth_np = depth.cpu().numpy()
                else:
                    depth_np = depth
                depth_show = depth_np[0] if depth_np.ndim == 3 else depth_np
                depth_show = (depth_show - depth_show.min()) / (depth_show.max() - depth_show.min() + 1e-8)
            else:
                depth_show = np.zeros_like(label_np, dtype=np.float32)

            # Column 0: GradCAM on fake_depth (predicted mask × depth)
            ax0 = fig.add_subplot(gs[row, 0])
            overlay_fake = overlay_cam_on_image(image_show.transpose(2, 0, 1), cam_fake_resized)
            ax0.imshow(overlay_fake)
            fake_color = "green" if logit_fake > 0 else "red"
            ax0.set_title(f"GradCAM(fake)\nlogit={logit_fake:.3f}", fontsize=9, color=fake_color)
            ax0.set_ylabel(f"S{sample_idx}", fontsize=9, fontweight="bold")
            ax0.set_xticks([])
            ax0.set_yticks([])

            # Column 1: GradCAM on real_depth (GT mask × depth)
            ax1 = fig.add_subplot(gs[row, 1])
            overlay_real = overlay_cam_on_image(image_show.transpose(2, 0, 1), cam_real_resized)
            ax1.imshow(overlay_real)
            real_color = "green" if logit_real > 0 else "red"
            ax1.set_title(f"GradCAM(real)\nlogit={logit_real:.3f}", fontsize=9, color=real_color)
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Column 2: Depth
            ax2 = fig.add_subplot(gs[row, 2])
            ax2.imshow(depth_show, cmap="gray")
            ax2.set_title("Depth" if row == 0 else "", fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Column 3: Predicted mask
            ax3 = fig.add_subplot(gs[row, 3])
            ax3.imshow(pred_mask, cmap="tab20", vmin=0, vmax=args.num_classes - 1)
            ax3.set_title("Pred Mask" if row == 0 else "", fontsize=10)
            ax3.set_xticks([])
            ax3.set_yticks([])

            # Column 4: Ground truth label
            ax4 = fig.add_subplot(gs[row, 4])
            ax4.imshow(label_np, cmap="tab20", vmin=0, vmax=args.num_classes - 1)
            ax4.set_title("GT Label" if row == 0 else "", fontsize=10)
            ax4.set_xticks([])
            ax4.set_yticks([])

            print(f"  [{img_idx+1}/{num_images}] Sample {sample_idx} done")

        save_path = os.path.join(args.output_dir, f"gradcam_disc_{img_idx+1}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
