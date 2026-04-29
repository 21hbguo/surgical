import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


TASK1_COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 255, 0),
}

TASK2_COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 128, 255),
    3: (255, 192, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 255, 0),
    7: (255, 128, 128),
}

TASK3_COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 64, 64),
    2: (64, 160, 255),
    3: (255, 224, 64),
    4: (160, 96, 255),
    5: (64, 224, 160),
    6: (255, 160, 64),
    7: (224, 64, 192),
}

@dataclass(frozen=True)
class DatasetLayout:
    image_column: str
    overlay_columns: Tuple[str, ...]


ENDOVIS2017_LAYOUT = DatasetLayout(
    image_column="images",
    overlay_columns=("labels_task3_class", "labels_task2_part", "labels_task1_binary"),
)
ENDOVIS2018_LAYOUT = DatasetLayout(
    image_column="images",
    overlay_columns=("labels_task2_class", "labels_task1_binary"),
)
DEFAULT_LAYOUT = ENDOVIS2017_LAYOUT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build EndoVis2017 overlay panels from matching filenames."
    )
    parser.add_argument(
        "--base-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data",
        help="Dataset root containing images and task label directories.",
    )
    parser.add_argument(
        "--layout",
        choices=["endovis2017", "endovis2018"],
        default="endovis2017",
        help="Overlay column layout to use.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/concat",
        help="Directory for stitched overlay panels.",
    )
    parser.add_argument(
        "--rows-per-panel",
        type=int,
        default=16,
        help="Number of samples per output panel.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=320,
        help="Per-overlay width after resizing.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=256,
        help="Per-overlay height after resizing.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0, 1].",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Maximum number of aligned samples to export; values <= 0 mean all samples.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260320,
        help="Fixed seed for deterministic sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker count; 0 means min(cpu_count, 8).",
    )
    return parser.parse_args()


def colorize_label(label: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_id, color in color_map.items():
        rgb[label == label_id] = color
    return rgb


def alpha_blend(image: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    blended = image.astype(np.float32).copy()
    mask_pixels = np.any(color_mask != 0, axis=2)
    blended[mask_pixels] = (
        (1.0 - alpha) * blended[mask_pixels] + alpha * color_mask[mask_pixels].astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def chunk_filenames(filenames: Sequence[str], group_size: int) -> List[List[str]]:
    return [list(filenames[idx : idx + group_size]) for idx in range(0, len(filenames), group_size)]


def sample_filenames(
    filenames: Sequence[str],
    sample_count: int,
    sample_seed: int,
) -> List[str]:
    ordered = list(filenames)
    if sample_count <= 0 or len(ordered) <= sample_count:
        return ordered
    rng = random.Random(sample_seed)
    sampled = rng.sample(ordered, sample_count)
    return sorted(sampled)


def _load_rgb_image(path: Path) -> np.ndarray:
    array = np.array(Image.open(path))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[..., :3]
    return array.astype(np.uint8)


def _load_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def _color_map_for_column(column_name: str) -> Dict[int, Tuple[int, int, int]]:
    if column_name == "labels_task1_binary":
        return TASK1_COLOR_MAP
    if column_name in {"labels_task2_part", "labels_task2_class"}:
        return TASK2_COLOR_MAP
    if column_name == "labels_task3_class":
        return TASK3_COLOR_MAP
    return TASK1_COLOR_MAP


def _resize_rgb(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(target_size, Image.Resampling.BILINEAR))


def build_panel_image(
    sample_paths: Sequence[Dict[str, Path]],
    target_size: Tuple[int, int],
    alpha: float = 0.45,
    image_column: str = DEFAULT_LAYOUT.image_column,
    overlay_columns: Sequence[str] = DEFAULT_LAYOUT.overlay_columns,
) -> np.ndarray:
    rows = []
    for sample_row in sample_paths:
        image = _load_rgb_image(sample_row[image_column])
        row_images = []
        for column_name in overlay_columns:
            color_mask = colorize_label(_load_label(sample_row[column_name]), _color_map_for_column(column_name))
            overlay = alpha_blend(image, color_mask, alpha=alpha)
            row_images.append(_resize_rgb(overlay, target_size))
        rows.append(np.concatenate(row_images, axis=1))
    return np.concatenate(rows, axis=0)


def collect_common_filenames(base_dir: Path, layout: DatasetLayout = DEFAULT_LAYOUT) -> List[str]:
    file_sets = []
    for column_name in [layout.image_column, *layout.overlay_columns]:
        directory = base_dir / column_name
        file_sets.append({path.name for path in directory.iterdir() if path.is_file()})
    return sorted(set.intersection(*file_sets))


def _build_sample_rows(
    base_dir: Path,
    filenames: Sequence[str],
    layout: DatasetLayout = DEFAULT_LAYOUT,
) -> List[Dict[str, Path]]:
    columns = [layout.image_column, *layout.overlay_columns]
    return [{column_name: base_dir / column_name / filename for column_name in columns} for filename in filenames]


def render_panel(
    base_dir: str,
    output_dir: str,
    filenames: Sequence[str],
    panel_index: int,
    target_size: Tuple[int, int],
    alpha: float,
    layout_name: str,
):
    base_path = Path(base_dir)
    out_dir = Path(output_dir)
    layout = _layout_from_name(layout_name)
    sample_rows = _build_sample_rows(base_path, filenames, layout=layout)
    panel_image = build_panel_image(
        sample_rows,
        target_size=target_size,
        alpha=alpha,
        image_column=layout.image_column,
        overlay_columns=layout.overlay_columns,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel_image).save(out_dir / f"panel_{panel_index:04d}.png")


def _run_panel_job(job):
    render_panel(*job)


def _layout_from_name(layout_name: str) -> DatasetLayout:
    if layout_name == "endovis2017":
        return ENDOVIS2017_LAYOUT
    if layout_name == "endovis2018":
        return ENDOVIS2018_LAYOUT
    return ENDOVIS2017_LAYOUT


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    layout = _layout_from_name(args.layout)
    filenames = collect_common_filenames(base_dir, layout=layout)
    filenames = sample_filenames(filenames, sample_count=args.sample_count, sample_seed=args.sample_seed)
    grouped_filenames = chunk_filenames(filenames, args.rows_per_panel)
    worker_count = args.workers or min(os.cpu_count() or 1, 8)
    target_size = (args.tile_width, args.tile_height)

    jobs = [
        (str(base_dir), str(output_dir), group, idx, target_size, args.alpha, args.layout)
        for idx, group in enumerate(grouped_filenames)
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(_run_panel_job, jobs))

    print(f"Saved {len(grouped_filenames)} panels to {output_dir}")


if __name__ == "__main__":
    main()
