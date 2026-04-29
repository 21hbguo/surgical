#!/usr/bin/env python3
"""Visualize EndoVis2017 class-wise domain gap using pretrained ResNet encoder features.

For each task (task1/task2/task3):
- Extract encoder final-stage feature map from images.
- Resize task label to feature-map resolution.
- Build per-class feature points by masked spatial average.
- Project points with t-SNE and draw one subplot per task.

Three task subplots are saved into one figure.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import Tensor
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.networks.block.resnetunet_block import ResNetEncoder
from utils.common import get_task_label_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TASKS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot task1/task2/task3 class-wise domain gap using pretrained ResNet encoder."
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/guo/project/ssl4mis/data/endovis2017",
        help="EndoVis2017 root path.",
    )
    parser.add_argument(
        "--pretrain_root",
        type=str,
        default="/home/guo/project/ssl4mis/pre_train_ckp",
        help="Pretrained weight root for ResNetEncoder.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34"],
        help="ResNet encoder variant.",
    )
    parser.add_argument(
        "--split_list",
        type=str,
        default="test_slices.list",
        help="Sample list file name under root_path, e.g., test_slices.list.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=300,
        help="Max images sampled from split list.",
    )
    parser.add_argument(
        "--max_points_per_class",
        type=int,
        default=400,
        help="Max feature points kept per class for each task.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input resize size for encoder.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/guo/project/ssl4mis/code_all/markdown/endovis2017_encoder_domain_gap_tasks123.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--exclude_background",
        action="store_true",
        help="Exclude class 0(background) from visualization.",
    )
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(images_dir: str, case_id: str) -> str:
    stem = os.path.join(images_dir, case_id)
    for ext in IMAGE_EXTENSIONS:
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return ""


def load_case_ids(root_path: str, split_list: str, max_samples: int, seed: int) -> List[str]:
    list_path = os.path.join(root_path, split_list)
    with open(list_path, "r", encoding="utf-8") as f:
        case_ids = [line.strip() for line in f if line.strip()]

    if split_list.startswith("test"):
        case_ids = [case_id.split(".")[0] for case_id in case_ids]

    if max_samples > 0 and len(case_ids) > max_samples:
        rng = random.Random(seed)
        case_ids = sorted(rng.sample(case_ids, max_samples))
    return case_ids


def load_task_classes(root_path: str, task: int) -> Dict[int, str]:
    json_path = os.path.join(root_path, f"task{task}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = {}
    for item in meta.get("classes", []):
        class_id = int(item["label_id"])
        classes[class_id] = str(item.get("name", f"class_{class_id}"))
    return classes


def preprocess_image(image_path: str, img_size: int) -> Tensor:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    image_chw = np.transpose(image_rgb, (2, 0, 1))
    return torch.from_numpy(image_chw).unsqueeze(0)


def load_label(label_path: str) -> np.ndarray:
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    return label.astype(np.int32)


def masked_class_vectors(feature_map: Tensor, label: np.ndarray, include_background: bool) -> Dict[int, np.ndarray]:
    _, channels, feat_h, feat_w = feature_map.shape
    resized = cv2.resize(label, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
    fmap = feature_map.squeeze(0).detach().cpu().numpy()  # C,H,W

    vectors: Dict[int, np.ndarray] = {}
    class_ids = np.unique(resized)
    for class_id in class_ids:
        if class_id < 0:
            continue
        if not include_background and class_id == 0:
            continue
        mask = resized == class_id
        if mask.sum() == 0:
            continue
        vec = fmap[:, mask].mean(axis=1)
        if vec.shape[0] != channels:
            continue
        vectors[int(class_id)] = vec.astype(np.float32)
    return vectors


def collect_task_features(
    encoder: ResNetEncoder,
    device: torch.device,
    root_path: str,
    task: int,
    case_ids: List[str],
    img_size: int,
    max_points_per_class: int,
    include_background: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    images_dir = os.path.join(root_path, "data", "images")
    label_dir_name = get_task_label_dir(root_path, task)
    labels_dir = os.path.join(root_path, "data", label_dir_name)
    class_names = load_task_classes(root_path, task)

    feat_buckets: Dict[int, List[np.ndarray]] = defaultdict(list)

    encoder.eval()
    with torch.no_grad():
        for case_id in tqdm(case_ids, desc=f"task{task}", ncols=100):
            image_path = resolve_image_path(images_dir, case_id)
            label_path = resolve_image_path(labels_dir, case_id)
            if not image_path or not label_path:
                continue

            image_tensor = preprocess_image(image_path, img_size).to(device)
            label = load_label(label_path)

            features = encoder(image_tensor)[-1]
            vectors = masked_class_vectors(features, label, include_background=include_background)
            for class_id, vec in vectors.items():
                feat_buckets[class_id].append(vec)

    sampled_features = []
    sampled_labels = []
    rng = random.Random(0)
    for class_id, vec_list in sorted(feat_buckets.items()):
        if not vec_list:
            continue
        if len(vec_list) > max_points_per_class:
            vec_list = rng.sample(vec_list, max_points_per_class)
        sampled_features.extend(vec_list)
        sampled_labels.extend([class_id] * len(vec_list))

    if not sampled_features:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32), class_names

    return (
        np.stack(sampled_features).astype(np.float32),
        np.array(sampled_labels, dtype=np.int32),
        class_names,
    )


def run_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    if len(features) <= 2:
        return np.zeros((len(features), 2), dtype=np.float32)
    effective = min(perplexity, max(1.0, float(len(features) - 1)))
    reducer = TSNE(
        n_components=2,
        perplexity=effective,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return reducer.fit_transform(features).astype(np.float32)


def plot_three_tasks(
    task_embeddings: Dict[int, np.ndarray],
    task_labels: Dict[int, np.ndarray],
    task_names: Dict[int, Dict[int, str]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=160)

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        emb = task_embeddings[task]
        labels = task_labels[task]
        class_map = task_names[task]

        if len(emb) == 0:
            ax.set_title(f"Task {task} (no points)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        class_ids = sorted(np.unique(labels).tolist())
        cmap = plt.get_cmap("tab20", max(2, len(class_ids)))

        for color_idx, class_id in enumerate(class_ids):
            mask = labels == class_id
            class_name = class_map.get(int(class_id), f"class_{int(class_id)}")
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=12,
                alpha=0.7,
                color=cmap(color_idx),
                label=f"{class_name}({int(class_id)})",
                linewidths=0,
            )

        ax.set_title(f"Task {task}: class-domain gap")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("EndoVis2017 Encoder Feature Domain Gap (Task1/2/3)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ResNetEncoder(
        variant=args.variant,
        in_chns=3,
        pretrain_root=args.pretrain_root,
        load_pretrained=True,
    ).to(device)

    case_ids = load_case_ids(args.root_path, args.split_list, args.max_samples, args.seed)
    print(f"Loaded {len(case_ids)} cases from {args.split_list}")

    task_embeddings: Dict[int, np.ndarray] = {}
    task_labels: Dict[int, np.ndarray] = {}
    task_names: Dict[int, Dict[int, str]] = {}

    for task in TASKS:
        features, labels, class_names = collect_task_features(
            encoder=encoder,
            device=device,
            root_path=args.root_path,
            task=task,
            case_ids=case_ids,
            img_size=args.img_size,
            max_points_per_class=args.max_points_per_class,
            include_background=not args.exclude_background,
        )
        print(f"Task {task}: points={len(features)}, classes={len(np.unique(labels)) if len(labels) else 0}")
        embedding = run_tsne(features, perplexity=args.perplexity, seed=args.seed)
        task_embeddings[task] = embedding
        task_labels[task] = labels
        task_names[task] = class_names

    plot_three_tasks(task_embeddings, task_labels, task_names, args.output)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
