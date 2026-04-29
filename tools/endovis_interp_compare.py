import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


DEFAULT_CASE = "seq_10_frame000"
DEFAULT_SIZE = (224, 224)
DEFAULT_ANGLE = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nearest-neighbor vs linear interpolation on one EndoVis2017 sample."
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help="Sample stem without extension, e.g. seq_10_frame000.",
    )
    parser.add_argument(
        "--base-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data",
        help="EndoVis2017 data directory.",
    )
    parser.add_argument(
        "--task-root",
        default="/home/guo/project/ssl4mis/data/endovis2017",
        help="Directory containing task1.json/task2.json/task3.json.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_SIZE[1],
        help="Target width after resize.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_SIZE[0],
        help="Target height after resize.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=DEFAULT_ANGLE,
        help="Rotation angle in degrees applied after resize.",
    )
    parser.add_argument(
        "--output",
        default="/home/guo/project/ssl4mis/data/endovis2017/interp_compare_seq_10_frame000.png",
        help="Output image path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def build_color_map(task_meta: dict) -> Dict[int, Tuple[int, int, int]]:
    classes = sorted(task_meta["classes"], key=lambda item: int(item["label_id"]))
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for idx, item in enumerate(classes):
        label_id = int(item["label_id"])
        if label_id == 0:
            color_map[label_id] = (0, 0, 0)
            continue
        hue = (idx * 0.16180339887498948) % 1.0
        saturation = 0.85
        value = 0.95
        color_map[label_id] = hsv_to_rgb(hue, saturation, value)
    return color_map


def load_image(path: Path) -> np.ndarray:
    image = np.array(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def load_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def colorize_label(label: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    canvas = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        canvas[label == label_id] = color
    return canvas


def resize_then_rotate(image: np.ndarray, target_size: Tuple[int, int], angle: float, order: int) -> np.ndarray:
    target_h, target_w = target_size
    if image.ndim == 2:
        pil_mode = "L"
    else:
        pil_mode = "RGB"
    pil_image = Image.fromarray(image, mode=pil_mode)
    resample = Image.Resampling.NEAREST if order == 0 else Image.Resampling.BILINEAR
    resized = np.array(pil_image.resize((target_w, target_h), resample=resample))
    rotated = ndimage.rotate(resized, angle=angle, order=order, reshape=False, mode="reflect")
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    if rotated.ndim == 2:
        rotated = np.repeat(rotated[..., None], 3, axis=2)
    return rotated


def resize_then_rotate_label_ids(
    label: np.ndarray,
    target_size: Tuple[int, int],
    angle: float,
    color_map: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    target_h, target_w = target_size
    pil_image = Image.fromarray(label, mode="L")
    resized = np.array(pil_image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)).astype(np.float32)
    rotated = ndimage.rotate(resized, angle=angle, order=1, reshape=False, mode="reflect")
    label_ids = np.rint(rotated).astype(np.int32)
    valid_ids = np.array(sorted(color_map.keys()), dtype=np.int32)
    label_ids = np.clip(label_ids, int(valid_ids.min()), int(valid_ids.max()))
    remapped = np.zeros_like(label_ids, dtype=np.uint8)
    for value in valid_ids:
        remapped[label_ids == int(value)] = np.uint8(value)
    return colorize_label(remapped, color_map)


def collect_case_paths(base_dir: Path, case_name: str) -> List[Tuple[str, Path]]:
    ordered = [
        ("depth1c", base_dir / "depth1c_slices" / f"{case_name}.png"),
        ("depth3c", base_dir / "depth3c_slices" / f"{case_name}.png"),
        ("image", base_dir / "images" / f"{case_name}.png"),
        ("task1", base_dir / "labels_task1_binary" / f"{case_name}.png"),
        ("task2", base_dir / "labels_task2_part" / f"{case_name}.png"),
        ("task3", base_dir / "labels_task3_class" / f"{case_name}.png"),
    ]
    missing = [str(path) for _, path in ordered if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for {case_name}: {missing}")
    return ordered


def render_panel(args: argparse.Namespace) -> np.ndarray:
    base_dir = Path(args.base_dir)
    task_root = Path(args.task_root)
    target_size = (args.height, args.width)

    task1_colors = build_color_map(load_json(task_root / "task1.json"))
    task2_colors = build_color_map(load_json(task_root / "task2.json"))
    task3_colors = build_color_map(load_json(task_root / "task3.json"))
    color_maps = {
        "task1": task1_colors,
        "task2": task2_colors,
        "task3": task3_colors,
    }

    columns = collect_case_paths(base_dir, args.case)
    rows = []
    for row_name, interp_order in [("nearest", 0), ("linear_rgb", 1)]:
        del row_name
        tiles = []
        for column_name, path in columns:
            if column_name in color_maps:
                source = colorize_label(load_label(path), color_maps[column_name])
            else:
                source = load_image(path)
            tiles.append(resize_then_rotate(source, target_size, args.angle, interp_order))
        rows.append(np.concatenate(tiles, axis=1))

    tiles = []
    for column_name, path in columns:
        if column_name in color_maps:
            source = resize_then_rotate_label_ids(
                load_label(path),
                target_size,
                args.angle,
                color_maps[column_name],
            )
        else:
            source = resize_then_rotate(load_image(path), target_size, args.angle, 1)
        tiles.append(source)
    rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    panel = render_panel(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(output_path)
    print(f"saved: {output_path}")
    print(f"case: {args.case}")
    print(f"size: {args.width}x{args.height}, angle: {args.angle}")


if __name__ == "__main__":
    main()
