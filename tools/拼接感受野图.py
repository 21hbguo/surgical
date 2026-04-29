
# 脚本工具：执行 拼接感受野图 相关的数据或分析任务。

import argparse
import importlib
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


# 解析训练脚本的命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Stitch receptive field visualization images into one figure.")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths in display order.")
    parser.add_argument("--titles", nargs="+", required=True, help="Column titles matching --images.")
    parser.add_argument("--save-path", required=True, help="Output path for the stitched figure.")
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Stitch direction.",
    )
    return parser.parse_args()
# 组织脚本主流程。


def main():
    args = parse_args()

    images = [np.asarray(Image.open(path).convert("RGB")) for path in args.images]
    if args.layout == "vertical":
        fig, axes = plt.subplots(len(images), 1, figsize=(8, 3.2 * len(images)))
    else:
        fig, axes = plt.subplots(1, len(images), figsize=(4.5 * len(images), 4.5))
    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, args.titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.03, wspace=0.03)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Stitched figure saved to: {args.save_path}")


if __name__ == "__main__":
    main()
