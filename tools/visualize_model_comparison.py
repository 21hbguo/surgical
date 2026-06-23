#!/usr/bin/env python3
"""拼接多个模型的预测结果对比图。
第一列为 label，后续每列为一个模型的预测（裁剪右半部分）。
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str,
                        default="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/task1")
    parser.add_argument("--label_dir", type=str,
                        default="/home/guo/project/ssl4mis/data/endovis2017/data/labels_task1_binary")
    parser.add_argument("--output_dir", type=str,
                        default="/home/guo/project/ssl4mis/result_predict/endovis2017_255_Samplinginterval/test")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--fold", type=str, default="f0")
    parser.add_argument("--models", type=str, nargs="*",
                        default=["Fully", "CPS", "MT", "MT_depth_guider_v4", "UAMT", "URPC", "U2PL_official", "UniMatch_official", "CorrMatch_official", "SegMatch_official", "GeoRiskSPC_DGv4"])
    parser.add_argument("--folds", type=str, nargs="*", default=["f0", "f1", "f2", "f3"])
    return parser.parse_args()


def find_model_subdir(model_dir):
    """找到模型的实际子目录（如 5_labeled_lr1e-4_s_unet）"""
    subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    return subdirs[0] if subdirs else None


def load_image(path):
    return Image.open(path).convert("RGB")


def get_font(size=12):
    """尝试加载字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for fold in args.folds:
        print(f"Processing {fold}...")
        # 获取样本列表
        fully_rgb_dir = os.path.join(args.task_dir, "Fully", find_model_subdir(os.path.join(args.task_dir, "Fully")), fold, "rgb")
        if not os.path.exists(fully_rgb_dir):
            print(f"  Skip {fold}: directory not found")
            continue
        all_images = sorted([f for f in os.listdir(fully_rgb_dir) if f.endswith(".png")])
        if args.num_samples > 0 and args.num_samples < len(all_images):
            indices = np.linspace(0, len(all_images) - 1, args.num_samples, dtype=int)
            selected_images = [all_images[i] for i in indices]
        else:
            selected_images = all_images

        # 模型列表
        models = args.models
        n_samples = len(selected_images)
        n_models = len(models)

        # 读取第一张图获取尺寸
        sample_img = load_image(os.path.join(fully_rgb_dir, selected_images[0]))
        w, h = sample_img.size

        # 布局参数
        cell_w = w
        cell_h = h
        row_title_h = 30  # 行标题高度
        col_title_h = 25  # 列标题高度
        gap = 4  # 间距

        # 创建画布
        total_w = (n_models + 1) * cell_w + (n_models + 1) * gap
        total_h = col_title_h + n_samples * (cell_h + gap) + row_title_h + gap
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        draw = ImageDraw.Draw(canvas)
        font_title = get_font(14)
        font_label = get_font(11)

        # 列标题
        title_map = {
            "Fully": "Fully",
            "CPS": "CPS",
            "MT": "MT",
            "MT_depth_guider_v4": "MT+DG",
            "UAMT": "UAMT",
            "URPC": "URPC",
            "U2PL_official": "U2PL",
            "UniMatch_official": "UniMatch",
            "CorrMatch_official": "CorrMatch",
            "SegMatch_official": "SegMatch",
            "GeoRiskSPC_DGv4": "GeoRisk",
        }
        col_titles = ["Label"] + [title_map.get(m, m) for m in models]
        for col_idx, title in enumerate(col_titles):
            x = gap + col_idx * (cell_w + gap)
            bbox = draw.textbbox((0, 0), title, font=font_title)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (cell_w - text_w) // 2, 2), title, fill=(0, 0, 0), font=font_title)

        # 填充图片
        for row_idx, img_name in enumerate(selected_images):
            # 行标题（样本名）
            seq_frame = img_name.replace(".png", "")
            y_row = col_title_h + row_idx * (cell_h + gap)
            draw.text((2, y_row + 2), seq_frame, fill=(100, 100, 100), font=font_label)

            # Label 列
            label_path = os.path.join(args.label_dir, img_name)
            if os.path.exists(label_path):
                label_img = Image.open(label_path).convert("L")
                # 二值图 (0/1) 映射到 (0/255)
                label_arr = np.array(label_img)
                if label_arr.max() <= 1:
                    label_arr = (label_arr * 255).astype(np.uint8)
                label_img = Image.fromarray(label_arr).resize((cell_w, cell_h), Image.NEAREST)
                x = gap
                canvas.paste(label_img, (x, y_row + row_title_h))

            # 模型列
            for col_idx, model in enumerate(models):
                subdir = find_model_subdir(os.path.join(args.task_dir, model))
                if subdir is None:
                    continue
                img_path = os.path.join(args.task_dir, model, subdir, fold, "rgb", img_name)
                if not os.path.exists(img_path):
                    continue
                img = load_image(img_path).resize((cell_w, cell_h), Image.NEAREST)
                x = gap + (col_idx + 1) * (cell_w + gap)
                canvas.paste(img, (x, y_row + row_title_h))

        # 保存
        output_path = os.path.join(args.output_dir, f"model_comparison_{fold}.png")
        canvas.save(output_path, dpi=(150, 150))
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
