
# 脚本工具：执行 域解释 相关的数据或分析任务。

import importlib
import os
import random
import sys
from itertools import combinations

import matplotlib
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

matplotlib.use('Agg')
plt = importlib.import_module("matplotlib.pyplot")

DOMAINS = [
    'endovis17',
    'endovis18',
    'kvasir_seg',
]
RGB_DIRS = [
    r'/home/guo/project/ssl4mis/data/endovis2017/data/images_slices',
    r'/home/guo/project/ssl4mis/data/endovis2018Binary8/data/images_slices',
    r'/home/guo/project/ssl4mis/data/kvasir_SEG/data/images_slices',
]
DEPTH_DIRS = [
    r'/home/guo/project/ssl4mis/data/endovis2017/data/depth3c_slices',
    r'/home/guo/project/ssl4mis/data/endovis2018Binary8/data/depth3c_slices',
    r'/home/guo/project/ssl4mis/data/kvasir_SEG/data/depth3c_slices',
]

MAX_SAMPLES = 1000
BATCH_SIZE = 32
PERPLEXITY = 30
OUTPUT_DIR = r"/home/guo/project/ssl4mis/code_all/claude/output"
SEED = 42
DINOV2_REPO = r"/home/guo/project/other_method/SSL/dinov2"
DINOV3_REPO = r"/home/guo/project/other_method/SSL/dinov3"
RESNET34_WEIGHTS = r"/home/guo/project/ssl4mis/pre_train_ckp/resnet34-b627a593.pth"
RESNET50_WEIGHTS = r"/home/guo/project/ssl4mis/pre_train_ckp/resnet50-11ad3fa6.pth"
DINOV2_WEIGHTS = r"/home/guo/project/ssl4mis/pre_train_ckp/dinov2_small.pth"
DINOV3_WEIGHTS = r"/home/guo/project/ssl4mis/pre_train_ckp/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

if DINOV2_REPO not in sys.path:
    sys.path.insert(0, DINOV2_REPO)
if DINOV3_REPO not in sys.path:
    sys.path.insert(0, DINOV3_REPO)

dinov2_vits14 = importlib.import_module("dinov2.hub.backbones").dinov2_vits14
dinov3_vits16 = importlib.import_module("dinov3.hub.backbones").dinov3_vits16

MODEL_SPECS = [
    {'name': 'resnet50', 'builder': 'resnet50'},
    {'name': 'resnet34', 'builder': 'resnet34'},
    {'name': 'dinov2_vits14', 'builder': 'dinov2_vits14'},
    {'name': 'dinov3_vits16', 'builder': 'dinov3_vits16'},
]


# 构建图像映射。
def build_image_map(image_dir):
    image_map = {}
    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        stem, _ = os.path.splitext(file_name)
        image_map[stem] = os.path.join(image_dir, file_name)
# 数据集模块：组织 SyncImageDataset 对应的样本访问。
    return image_map

# 组织当前数据集的样本读取逻辑。
class SyncImageDataset(Dataset):
    # 初始化参数与子模块。
    def __init__(self, rgb_dir, depth_dir, max_samples=200, seed=42):
        rng = random.Random(seed)
        rgb_map = build_image_map(rgb_dir)
        depth_map = build_image_map(depth_dir)
        common_stems = sorted(set(rgb_map) & set(depth_map))
        if max_samples < len(common_stems):
            common_stems = rng.sample(common_stems, max_samples)
            common_stems.sort()
        self.rgb_paths = [rgb_map[stem] for stem in common_stems]
        self.depth_paths = [depth_map[stem] for stem in common_stems]
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
    # 返回当前数据规模。
        self.transform_rgb = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)])
        self.transform_depth = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)])
    # 返回当前数据规模。
    def __len__(self):
        return len(self.rgb_paths)
    # 按索引构造一个样本。
    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx]).convert('RGB')
        return self.transform_rgb(rgb), self.transform_depth(depth)

# 加载相关内容。
def load_resnet_backbone(model_name, weights_path):
    if model_name == "resnet34":
        backbone = models.resnet34(weights=None)
    else:
        backbone = models.resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=True)
    return backbone


# 封装当前模块的主要职责。
class RGBExtractor(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
    # 定义模块前向计算。
        backbone = load_resnet_backbone("resnet50", RESNET50_WEIGHTS)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
    # 定义模块前向计算。
    def forward(self, x):
        return self.features(x).flatten(1)

# 封装当前模块的主要职责。
class DepthExtractor(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
    # 定义模块前向计算。
        backbone = load_resnet_backbone("resnet50", RESNET50_WEIGHTS)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
    # 定义模块前向计算。
    def forward(self, x):
        return self.features(x).flatten(1)


# 封装当前模块的主要职责。
class RGBExtractor34(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
        backbone = load_resnet_backbone("resnet34", RESNET34_WEIGHTS)
    # 定义模块前向计算。
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    # 定义模块前向计算。
    def forward(self, x):
        return self.features(x).flatten(1)


# 封装当前模块的主要职责。
class DepthExtractor34(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
        backbone = load_resnet_backbone("resnet34", RESNET34_WEIGHTS)
    # 定义模块前向计算。
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    # 定义模块前向计算。
    def forward(self, x):
        return self.features(x).flatten(1)


# 封装当前模块的主要职责。
class DinoV2Extractor(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
    # 定义模块前向计算。
        self.backbone = dinov2_vits14(pretrained=True, weights=DINOV2_WEIGHTS)

    # 定义模块前向计算。
    def forward(self, x):
        return self.backbone(x)


# 封装当前模块的主要职责。
class DinoV3Extractor(nn.Module):
    # 初始化参数与子模块。
    def __init__(self):
        super().__init__()
    # 定义模块前向计算。
        self.backbone = dinov3_vits16(pretrained=True, weights=DINOV3_WEIGHTS)

    # 定义模块前向计算。
    def forward(self, x):
        return self.backbone(x)


# 构建相关内容。
def build_extractors(model_name):
    if model_name == 'resnet50':
        return RGBExtractor(), DepthExtractor()
    if model_name == 'resnet34':
        return RGBExtractor34(), DepthExtractor34()
    if model_name == 'dinov2_vits14':
        return DinoV2Extractor(), DinoV2Extractor()
    if model_name == 'dinov3_vits16':
        return DinoV3Extractor(), DinoV3Extractor()
    return RGBExtractor(), DepthExtractor()

# 计算相关内容。
def compute_mmd(feat_a, feat_b):
    all_feats = torch.cat([feat_a, feat_b], dim=0)
    pairwise_dist = torch.cdist(all_feats, all_feats)
    median_val = torch.median(pairwise_dist).item()
    gamma = 1.0 / (2 * median_val ** 2 + 1e-8)
    # 处理相关内容相关逻辑。
    def rbf(X, Y):
        return torch.exp(-gamma * torch.cdist(X, Y) ** 2)
    return (rbf(feat_a, feat_a).mean() + rbf(feat_b, feat_b).mean() - 2 * rbf(feat_a, feat_b).mean()).item()

# 提取相关内容。
def extract_features(rgb_dir, depth_dir, domain_name, rgb_extractor, depth_extractor, device, max_samples, batch_size):
    dataset = SyncImageDataset(rgb_dir, depth_dir, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    rgb_feats, depth_feats = [], []
    with torch.no_grad():
        for rgb_batch, depth_batch in loader:
            rgb_batch = rgb_batch.to(device)
            depth_batch = depth_batch.to(device)
            rgb_feats.append(rgb_extractor(rgb_batch).cpu())
            depth_feats.append(depth_extractor(depth_batch).cpu())
    print(f"  [{domain_name}] loaded {len(dataset)} samples")
    return torch.cat(rgb_feats), torch.cat(depth_feats)

# 计算相关内容。
def compute_joint_tsne(rgb_feats_dict, depth_feats_dict, perplexity=30):
    domains = list(rgb_feats_dict.keys())
    rgb_all_feats = np.vstack([f.numpy() for f in rgb_feats_dict.values()])
    depth_all_feats = np.vstack([f.numpy() for f in depth_feats_dict.values()])
    rgb_labels = np.concatenate([np.full(len(f), i) for i, f in enumerate(rgb_feats_dict.values())])
    depth_labels = np.concatenate([np.full(len(f), i) for i, f in enumerate(depth_feats_dict.values())])

    joint_feats = np.vstack([rgb_all_feats, depth_all_feats])
    joint_feats = StandardScaler().fit_transform(joint_feats)
    effective_perplexity = min(perplexity, max(1, len(joint_feats) - 1))
    joint_reduced = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        max_iter=1000,
    ).fit_transform(joint_feats)
    return {
        'domains': domains,
        'rgb_reduced': joint_reduced[:len(rgb_all_feats)],
        'depth_reduced': joint_reduced[len(rgb_all_feats):],
        'rgb_labels': rgb_labels,
        'depth_labels': depth_labels,
        'x_min': joint_reduced[:, 0].min(),
        'x_max': joint_reduced[:, 0].max(),
        'y_min': joint_reduced[:, 1].min(),
        'y_max': joint_reduced[:, 1].max(),
    }


# 处理相关内容相关逻辑。
def plot_tsne_grid(model_results, save_path):
    colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C','#E67E22','#34495E']
    model_names = list(model_results.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(7 * len(model_names), 12), squeeze=False)

    for col, model_name in enumerate(model_names):
        result = model_results[model_name]
        for row, (reduced, labels, row_title) in enumerate([
            (result['rgb_reduced'], result['rgb_labels'], 'RGB Features'),
            (result['depth_reduced'], result['depth_labels'], 'Depth Features'),
        ]):
            ax = axes[row, col]
            for i, domain in enumerate(result['domains']):
                mask = labels == i
                ax.scatter(reduced[mask, 0], reduced[mask, 1], c=colors[i % len(colors)], alpha=0.6, s=15, label=domain)
            ax.set_title(f'{model_name} - {row_title}', fontsize=13, fontweight='bold')
            ax.set_xlabel('t-SNE dim 1')
            ax.set_ylabel('t-SNE dim 2')
            ax.set_xlim(result['x_min'], result['x_max'])
            ax.set_ylim(result['y_min'], result['y_max'])
            if row == 0:
                ax.legend(fontsize=9, markerscale=2)

    plt.suptitle('Domain Gap Analysis Across Models (Joint t-SNE)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to: {save_path}")
    plt.close(fig)

# 执行相关内容。
def run_mmd_analysis(model_name, rgb_feats_dict, depth_feats_dict):
    domains = list(rgb_feats_dict.keys())
    print(f"\n[{model_name}]")
    print("="*60)
    print(f"{'Pair':<35} {'RGB MMD':>10} {'Depth MMD':>10} {'Reduction':>10}")
    print("="*60)
    rgb_mmds, depth_mmds = [], []
    for d_i, d_j in combinations(domains, 2):
        rgb_mmd = compute_mmd(rgb_feats_dict[d_i], rgb_feats_dict[d_j])
        depth_mmd = compute_mmd(depth_feats_dict[d_i], depth_feats_dict[d_j])
        reduction = (1 - depth_mmd / (rgb_mmd + 1e-8)) * 100
        rgb_mmds.append(rgb_mmd)
        depth_mmds.append(depth_mmd)
        pair_str = f"{d_i} vs {d_j}"
        print(f"{pair_str:<35} {rgb_mmd:>10.4f} {depth_mmd:>10.4f} {reduction:>9.1f}%")
    print("="*60)
    avg_rgb = np.mean(rgb_mmds)
    avg_depth = np.mean(depth_mmds)
    avg_reduction = (1 - avg_depth / (avg_rgb + 1e-8)) * 100
    print(f"{'Average':<35} {avg_rgb:>10.4f} {avg_depth:>10.4f} {avg_reduction:>9.1f}%")
# 组织脚本主流程。
    print("="*60)

# 组织脚本主流程。
def main():
    assert len(DOMAINS) == len(RGB_DIRS) == len(DEPTH_DIRS), \
        "DOMAINS, RGB_DIRS, DEPTH_DIRS must have the same number of entries"
    assert len(DOMAINS) >= 2, "Need at least 2 domains to compare"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    all_tsne_results = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nExtracting features...")
    for model_spec in MODEL_SPECS:
        model_name = model_spec['name']
        print(f"\n===== Running model: {model_name} =====")
        rgb_extractor, depth_extractor = build_extractors(model_name)
        rgb_extractor = rgb_extractor.to(device).eval()
        depth_extractor = depth_extractor.to(device).eval()
        rgb_feats_dict = {}
        depth_feats_dict = {}
        for name, rgb_dir, depth_dir in zip(DOMAINS, RGB_DIRS, DEPTH_DIRS):
            rgb_f, depth_f = extract_features(
                rgb_dir, depth_dir, name, rgb_extractor, depth_extractor, device, MAX_SAMPLES, BATCH_SIZE
            )
            rgb_feats_dict[name] = rgb_f
            depth_feats_dict[name] = depth_f
        run_mmd_analysis(model_name, rgb_feats_dict, depth_feats_dict)
        print(f"[{model_name}] Running t-SNE (may take a few minutes)...")
        all_tsne_results[model_name] = compute_joint_tsne(rgb_feats_dict, depth_feats_dict, PERPLEXITY)
        del rgb_extractor
        del depth_extractor
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    tsne_path = os.path.join(OUTPUT_DIR, 'domain_gap_tsne_all_models.pdf')
    plot_tsne_grid(all_tsne_results, tsne_path)

if __name__ == '__main__':
    main()
