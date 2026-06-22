import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image

project_dir = Path(__file__).resolve().parents[1]
ssl4mis_dir = project_dir.parent
sam3_dir = ssl4mis_dir.parent / "other_method" / "other" / "sam3"
transformers_src = sam3_dir / "transformers" / "src"
if transformers_src.exists():
    sys.path.insert(0, str(transformers_src))
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="/home/guo/project/ssl4mis/data/endovis2017_h5_224_224/data/images", help="input folder containing png, jpg, jpeg, h5, or hdf5 images")
parser.add_argument("--output", default="/home/guo/project/ssl4mis/data/endovis2017_h5_224_224/data/sam3_seg", help="output folder for RGB masks, label masks, and optional metadata")
parser.add_argument("--model", default=str(ssl4mis_dir / "pre_train_ckp" / "sam3_checkpoints"), help="local HuggingFace SAM3 folder or model id")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--points-per-batch", type=int, default=64)
parser.add_argument("--points-per-crop", type=int, default=32)
parser.add_argument("--crops-n-layers", type=int, default=0)
parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
parser.add_argument("--stability-score-thresh", type=float, default=0.95)
parser.add_argument("--mask-threshold", type=float, default=0.0)
parser.add_argument("--crops-nms-thresh", type=float, default=0.7)
parser.add_argument("--h5-scale", type=float, default=255.0, help="scale factor applied to floating point h5 images")
parser.add_argument("--save-meta", action="store_true")
parser.add_argument("--skip-existing", action="store_true")
args = parser.parse_args()
input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
paths = []
for suffix in ("*.png", "*.jpg", "*.jpeg", "*.h5", "*.hdf5"):
    paths.extend(input_dir.rglob(suffix))
paths = sorted(path for path in paths if path.is_file() and output_dir not in path.parents)
device = 0 if args.device.startswith("cuda") else args.device
generator = pipeline("mask-generation", model=args.model, device=device)
for index, path in enumerate(paths, 1):
    relative_dir = path.relative_to(input_dir).parent
    save_dir = output_dir / relative_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    mask_path = save_dir / f"{path.stem}.png"
    rgb_mask_path = save_dir / f"{path.stem}_mask.png"
    label_path = save_dir / f"{path.stem}_label.png"
    meta_path = save_dir / f"{path.stem}.json"
    if args.skip_existing and mask_path.exists():
        print(f"[{index}/{len(paths)}] skip {path.name}")
        continue
    if path.suffix.lower() in (".h5", ".hdf5"):
        selected = None
        with h5py.File(path, "r") as f:
            for key in ("image", "images", "img", "data"):
                if key in f and isinstance(f[key], h5py.Dataset):
                    selected = f[key][()]
                    break
            if selected is None:
                for key in f:
                    obj = f[key]
                    if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                        selected = obj[()]
                        break
        if selected is None:
            raise ValueError(f"no image-like dataset found in {path}")
        image_array = np.squeeze(np.asarray(selected))
        if image_array.ndim == 3 and image_array.shape[0] in (1, 3) and image_array.shape[-1] not in (1, 3):
            image_array = np.transpose(image_array, (1, 2, 0))
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=-1)
        if image_array.ndim == 3 and image_array.shape[-1] == 1:
            image_array = np.repeat(image_array, 3, axis=-1)
        if image_array.ndim != 3 or image_array.shape[-1] < 3:
            raise ValueError(f"unsupported image shape {image_array.shape} in {path}")
        image_array = image_array[..., :3]
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = image_array * args.h5_scale
        image_array = np.nan_to_num(image_array, nan=0.0, posinf=255.0, neginf=0.0)
        image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB")
    else:
        image = Image.open(path).convert("RGB")
    outputs = generator(
        image,
        points_per_batch=args.points_per_batch,
        points_per_crop=args.points_per_crop,
        crops_n_layers=args.crops_n_layers,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        mask_threshold=args.mask_threshold,
        crops_nms_thresh=args.crops_nms_thresh,
        output_bboxes_mask=True,
    )
    masks = outputs.get("masks", [])
    scores = outputs.get("scores", [])
    boxes = outputs.get("bounding_boxes", [])
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().tolist()
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().tolist()
    order = np.argsort(np.asarray(scores, dtype=np.float32)) if len(scores) else np.arange(len(masks))
    label = np.zeros((image.height, image.width), dtype=np.uint16)
    for instance_id, mask_index in enumerate(order, 1):
        mask = masks[int(mask_index)]
        if isinstance(mask, Image.Image):
            mask = np.asarray(mask)
        elif torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        else:
            mask = np.asarray(mask)
        mask = np.squeeze(mask).astype(bool)
        if mask.shape != label.shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((image.width, image.height), Image.NEAREST)).astype(bool)
        label[mask] = instance_id
    color_map = np.array([
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ], dtype=np.uint8)
    color = color_map[label % len(color_map)]
    image_array = np.asarray(image)
    concat = np.concatenate([image_array, color], axis=1)
    Image.fromarray(concat, mode="RGB").save(mask_path)
    Image.fromarray(color, mode="RGB").save(rgb_mask_path)
    Image.fromarray(label).save(label_path)
    if args.save_meta:
        meta = {
            "source": str(path),
            "image_size": [image.width, image.height],
            "num_masks": int(len(masks)),
            "scores": [float(scores[int(i)]) for i in order] if len(scores) else [],
            "boxes": [boxes[int(i)] for i in order] if len(boxes) else [],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{index}/{len(paths)}] {path.name} -> {mask_path.name} masks={len(masks)}")
