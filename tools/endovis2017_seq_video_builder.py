import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


FILENAME_PATTERN = re.compile(r"seq_(\d+)_frame(\d+)\.png$")
DEFAULT_INPUT_SIZE = 224
DEFAULT_ALPHA = 0.45
DEFAULT_FPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-sequence MP4 videos by horizontally stitching EndoVis2017 frames."
    )
    parser.add_argument(
        "--images-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/images",
        help="Directory containing RGB images.",
    )
    parser.add_argument(
        "--task1-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/labels_task1_binary",
        help="Directory containing task1 labels.",
    )
    parser.add_argument(
        "--task2-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/labels_task2_part",
        help="Directory containing task2 labels.",
    )
    parser.add_argument(
        "--task3-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/labels_task3_class",
        help="Directory containing task3 labels.",
    )
    parser.add_argument(
        "--depth-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/depth3c_slices",
        help="Directory containing 3-channel depth slices.",
    )
    parser.add_argument(
        "--task1-json",
        default="/home/guo/project/ssl4mis/data/endovis2017/task1.json",
        help="Task1 metadata json.",
    )
    parser.add_argument(
        "--task2-json",
        default="/home/guo/project/ssl4mis/data/endovis2017/task2.json",
        help="Task2 metadata json.",
    )
    parser.add_argument(
        "--task3-json",
        default="/home/guo/project/ssl4mis/data/endovis2017/task3.json",
        help="Task3 metadata json.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017_video",
        help="Output directory for seqXXX.mp4 files.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Resize each panel tile to size x size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Alpha used when blending label RGB overlays on the source image.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output video fps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker count. 0 means min(cpu_count, number of seqs).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_color_map(task_meta: dict) -> Dict[int, Tuple[int, int, int]]:
    classes = sorted(task_meta["classes"], key=lambda item: int(item["label_id"]))
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for idx, item in enumerate(classes):
        label_id = int(item["label_id"])
        if label_id == 0:
            color_map[label_id] = (0, 0, 0)
            continue
        hue = (idx * 0.16180339887498948) % 1.0
        saturation = 0.75 + 0.15 * ((idx % 3) / 2.0)
        value = 0.9
        rgb = hsv_to_rgb(hue, saturation, value)
        color_map[label_id] = rgb
    return color_map


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    rgb = cv2.cvtColor(
        np.array([[[int(h * 179), int(s * 255), int(v * 255)]]], dtype=np.uint8),
        cv2.COLOR_HSV2RGB,
    )[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def parse_frame_name(filename: str) -> Tuple[int, int]:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return int(match.group(1)), int(match.group(2))


def collect_sequences(*directories: Path) -> Dict[int, List[str]]:
    common_names = None
    for directory in directories:
        names = {path.name for path in directory.iterdir() if path.is_file() and FILENAME_PATTERN.match(path.name)}
        common_names = names if common_names is None else common_names & names
    if not common_names:
        return {}

    seq_to_names: Dict[int, List[str]] = {}
    for name in sorted(common_names, key=lambda item: parse_frame_name(item)):
        seq_id, _ = parse_frame_name(name)
        seq_to_names.setdefault(seq_id, []).append(name)
    return seq_to_names


def load_rgb(path: Path) -> np.ndarray:
    image = np.array(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    return image.astype(np.uint8)


def load_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR))


def resize_label(label: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(label).resize((size, size), Image.Resampling.NEAREST))


def colorize_label(label: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    canvas = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        canvas[label == label_id] = color
    return canvas


def overlay_mask(image: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    blended = image.astype(np.float32).copy()
    active = np.any(color_mask != 0, axis=2)
    blended[active] = (1.0 - alpha) * blended[active] + alpha * color_mask[active].astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_frame(
    name: str,
    images_dir: Path,
    task1_dir: Path,
    task2_dir: Path,
    task3_dir: Path,
    depth_dir: Path,
    task1_colors: Dict[int, Tuple[int, int, int]],
    task2_colors: Dict[int, Tuple[int, int, int]],
    task3_colors: Dict[int, Tuple[int, int, int]],
    size: int,
    alpha: float,
) -> np.ndarray:
    image = resize_rgb(load_rgb(images_dir / name), size)
    depth = resize_rgb(load_rgb(depth_dir / name), size)

    task1_label = resize_label(load_label(task1_dir / name), size)
    task2_label = resize_label(load_label(task2_dir / name), size)
    task3_label = resize_label(load_label(task3_dir / name), size)

    task1_overlay = overlay_mask(image, colorize_label(task1_label, task1_colors), alpha)
    task2_overlay = overlay_mask(image, colorize_label(task2_label, task2_colors), alpha)
    task3_overlay = overlay_mask(image, colorize_label(task3_label, task3_colors), alpha)

    frame = np.concatenate([image, task1_overlay, task2_overlay, task3_overlay, depth], axis=1)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def write_sequence_video(job: tuple) -> Tuple[int, int, str]:
    (
        seq_id,
        frame_names,
        images_dir,
        task1_dir,
        task2_dir,
        task3_dir,
        depth_dir,
        output_dir,
        task1_colors,
        task2_colors,
        task3_colors,
        size,
        alpha,
        fps,
    ) = job

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seq{seq_id:03d}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (size * 5, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    frame_count = 0
    try:
        for name in frame_names:
            frame = build_frame(
                name=name,
                images_dir=images_dir,
                task1_dir=task1_dir,
                task2_dir=task2_dir,
                task3_dir=task3_dir,
                depth_dir=depth_dir,
                task1_colors=task1_colors,
                task2_colors=task2_colors,
                task3_colors=task3_colors,
                size=size,
                alpha=alpha,
            )
            writer.write(frame)
            frame_count += 1
    finally:
        writer.release()

    return seq_id, frame_count, str(output_path)


def build_jobs(
    seq_to_names: Dict[int, List[str]],
    args: argparse.Namespace,
    task1_colors: Dict[int, Tuple[int, int, int]],
    task2_colors: Dict[int, Tuple[int, int, int]],
    task3_colors: Dict[int, Tuple[int, int, int]],
) -> Iterable[tuple]:
    for seq_id, frame_names in sorted(seq_to_names.items()):
        yield (
            seq_id,
            frame_names,
            Path(args.images_dir),
            Path(args.task1_dir),
            Path(args.task2_dir),
            Path(args.task3_dir),
            Path(args.depth_dir),
            Path(args.output_dir),
            task1_colors,
            task2_colors,
            task3_colors,
            args.size,
            args.alpha,
            args.fps,
        )


def main() -> None:
    args = parse_args()
    task1_meta = load_json(Path(args.task1_json))
    task2_meta = load_json(Path(args.task2_json))
    task3_meta = load_json(Path(args.task3_json))

    task1_colors = build_color_map(task1_meta)
    task2_colors = build_color_map(task2_meta)
    task3_colors = build_color_map(task3_meta)

    seq_to_names = collect_sequences(
        Path(args.images_dir),
        Path(args.task1_dir),
        Path(args.task2_dir),
        Path(args.task3_dir),
        Path(args.depth_dir),
    )
    if not seq_to_names:
        raise RuntimeError("No aligned frames found across the five input directories.")

    jobs = list(build_jobs(seq_to_names, args, task1_colors, task2_colors, task3_colors))
    worker_count = args.workers or min(mp.cpu_count(), len(jobs))
    worker_count = max(worker_count, 1)

    if worker_count == 1:
        results = [write_sequence_video(job) for job in jobs]
    else:
        with mp.Pool(processes=worker_count) as pool:
            results = pool.map(write_sequence_video, jobs)

    for seq_id, frame_count, output_path in sorted(results):
        print(f"seq{seq_id:03d}: {frame_count} frames -> {output_path}")


if __name__ == "__main__":
    main()
