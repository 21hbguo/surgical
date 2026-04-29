import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


ALLOWED_VALUES = {0, 30, 100, 255}


def create_problem_mask(label: np.ndarray) -> np.ndarray:
    return np.where(np.isin(label, list(ALLOWED_VALUES)), 0, 255).astype(np.uint8)


def create_overlay_image(image: np.ndarray, problem_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    problem_pixels = problem_mask > 0
    highlight = np.zeros_like(overlay)
    highlight[..., 0] = 255
    overlay[problem_pixels] = (1.0 - alpha) * overlay[problem_pixels] + alpha * highlight[problem_pixels]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_output_name(dataset_dir: Path, image_path: Path) -> str:
    seq_name = dataset_dir.name.replace("instrument_dataset_", "seq_")
    return f"{seq_name}_{image_path.name}"


def find_problem_files(root_dir: Path) -> List[Tuple[Path, Path]]:
    problem_files: List[Tuple[Path, Path]] = []
    for dataset_dir in sorted(root_dir.glob("instrument_dataset_*")):
        parts_dir = dataset_dir / "ground_truth" / "PartsSegmentation"
        if not parts_dir.is_dir():
            continue
        for image_path in sorted(parts_dir.iterdir()):
            if not image_path.is_file():
                continue
            label = np.array(Image.open(image_path))
            if np.any(~np.isin(label, list(ALLOWED_VALUES))):
                problem_files.append((dataset_dir, image_path))
    return problem_files


def export_problem_masks(root_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_files = find_problem_files(root_dir)
    for dataset_dir, image_path in problem_files:
        label = np.array(Image.open(image_path))
        problem_mask = create_problem_mask(label)
        left_frame_path = dataset_dir / "left_frames" / image_path.name
        image = np.array(Image.open(left_frame_path).convert("RGB"))
        overlay = create_overlay_image(image, problem_mask)
        Image.fromarray(overlay).save(output_dir / build_output_name(dataset_dir, image_path))
    return len(problem_files)


def parse_args():
    parser = argparse.ArgumentParser(description="Export binary masks for anomalous PartsSegmentation labels.")
    parser.add_argument(
        "--root-dir",
        default="/home/guo/project/data/original_data/endovissub2017-robotics/instrument_2017_test",
        help="Root directory containing instrument_dataset_*/ground_truth/PartsSegmentation.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/guo/project/data/original_data/endovissub2017-robotics/test_problem",
        help="Directory for exported problem overlays.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    count = export_problem_masks(Path(args.root_dir), Path(args.output_dir))
    print(f"Exported {count} problem masks to {args.output_dir}")


if __name__ == "__main__":
    main()
