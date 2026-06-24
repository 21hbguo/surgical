"""
拼图脚本：每张图10行，每行一个样本，列=[Image, Label, FullyDepthGAN, MT_depth_guider_v4]
"""
import os
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 配置
BASE = Path("/home/guo/project/ssl4mis/result_predict/endovis2017_h5_224_224_255_Samplinginterval/task1")
IMAGE_DIR = Path("/home/guo/project/ssl4mis/data/endovis2017_h5_224_224/data/images")
LABEL_DIR = Path("/home/guo/project/ssl4mis/data/endovis2017_h5_224_224/data/labels_task1_binary")
OUTPUT_DIR = Path("/home/guo/project/ssl4mis/data/test2")

# 预测目录
PRED_DIRS = [
    (BASE / "Fully" / "100_labeled_lr1e-4_unet", "Fully"),
    (BASE / "FullyDepthGAN_ganLR1e-5" / "100_labeled_lr1e-4_unet_depth1C", "FullyDepthGAN"),
    (BASE / "MT_depth_guider_v4_gan" / "40_labeled_lr1e-4_s_unet_depth_guider_v4_depth1C", "MT_depth_guider"),
]
COL_NAMES = ["Image", "Label"] + [c[1] for c in PRED_DIRS]
SAMPLES_PER_ROW = 10

# 标签颜色映射
LABEL_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
}


def h5_to_rgb(h5_path):
    with h5py.File(h5_path, 'r') as f:
        mask = f['img'][0]
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in LABEL_COLORS.items():
        rgb[mask == val] = color
    return Image.fromarray(rgb)


def h5_to_image(h5_path):
    with h5py.File(h5_path, 'r') as f:
        img = f['img'][()]
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def get_all_images(pred_dir):
    images = {}
    for fold_name in ['f0', 'f1']:
        sub = pred_dir / fold_name
        if sub.is_dir():
            rgb_dir = sub / 'rgb'
            if rgb_dir.exists():
                for img in sorted(rgb_dir.glob('*.png')):
                    images[img.stem] = img
    return images


def make_overlay(pred_path, label_path):
    """生成TP=绿, FN=红, FP=蓝的叠加图"""
    with h5py.File(label_path, 'r') as f:
        label = f['img'][0]
    pred_img = np.array(Image.open(pred_path))
    pred = (pred_img[:, :, 0] > 128).astype(np.uint8)  # 红色通道提取预测mask
    label = (label > 0).astype(np.uint8)

    h, w = label.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[(label == 1) & (pred == 1)] = [0, 200, 0]    # TP=绿
    overlay[(label == 1) & (pred == 0)] = [200, 0, 0]    # FN=红
    overlay[(label == 0) & (pred == 1)] = [0, 0, 200]    # FP=蓝
    return Image.fromarray(overlay)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_preds = [get_all_images(d[0]) for d in PRED_DIRS]

    # 取所有预测目录的交集，只保留共同样本
    common_keys = set(all_preds[0].keys())
    for pred_dict in all_preds[1:]:
        common_keys &= set(pred_dict.keys())
    common_keys = sorted(common_keys)
    num_samples = len(common_keys)
    print(f"Total common samples: {num_samples}")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    sample_img = Image.open(next(iter(all_preds[0].values())))
    img_w, img_h = sample_img.size
    del sample_img

    n_cols = len(COL_NAMES)
    gap = 4
    header_h = 30
    row_gap = 4

    total_w = n_cols * img_w + (n_cols - 1) * gap
    total_h = header_h + SAMPLES_PER_ROW * (img_h + row_gap)

    num_mosaics = (num_samples + SAMPLES_PER_ROW - 1) // SAMPLES_PER_ROW
    print(f"Creating {num_mosaics} mosaics, {SAMPLES_PER_ROW} samples each")

    for mosaic_idx in range(num_mosaics):
        canvas = Image.new('RGB', (total_w, total_h), 'white')
        draw = ImageDraw.Draw(canvas)

        # 列标题
        for col_idx, col_name in enumerate(COL_NAMES):
            x = col_idx * (img_w + gap)
            bbox = draw.textbbox((0, 0), col_name, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (img_w - tw) // 2, 5), col_name, fill='black', font=font)

        # 绘制每行
        for local_idx in range(SAMPLES_PER_ROW):
            global_idx = mosaic_idx * SAMPLES_PER_ROW + local_idx
            if global_idx >= num_samples:
                break

            y = header_h + local_idx * (img_h + row_gap)
            frame_name = common_keys[global_idx]

            # Image 列
            img_path = IMAGE_DIR / f"{frame_name}.h5"
            if img_path.exists():
                canvas.paste(h5_to_image(img_path), (0, y))

            # Label 列
            label_path = LABEL_DIR / f"{frame_name}.h5"
            if label_path.exists():
                canvas.paste(h5_to_rgb(label_path), (1 * (img_w + gap), y))

            # 预测列（叠加图）
            if label_path.exists():
                for col_idx, pred_dict in enumerate(all_preds):
                    if frame_name in pred_dict:
                        canvas.paste(make_overlay(pred_dict[frame_name], label_path),
                                     ((col_idx + 2) * (img_w + gap), y))

            # 行标签
            short_name = frame_name.replace('seq_', 's').replace('_frame', 'f')
            draw.text((total_w + 5, y + img_h // 2 - 8), short_name, fill='gray', font=font_small)

        out_path = OUTPUT_DIR / f"mosaic_{mosaic_idx:03d}.png"
        canvas.save(out_path)
        print(f"Saved: {out_path}")

    print(f"Done!")


if __name__ == "__main__":
    main()
