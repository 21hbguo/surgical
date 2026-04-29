import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 128, 255),
    3: (255, 192, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 255, 0),
    7: (255, 128, 128),
    8: (255, 255, 255),
    255: (255, 255, 255),
}


def remap_label(label: np.ndarray, source_value: int, target_value: int) -> np.ndarray:
    remapped = label.copy()
    remapped[remapped == source_value] = target_value
    return remapped.astype(np.uint8)


def remap_directory(
    src_dir: Path,
    source_value: int,
    target_value: int,
    *,
    dst_dir: Path | None = None,
    inplace: bool = False,
    backup_dir: Path | None = None,
) -> int:
    if inplace and backup_dir is None:
        raise ValueError("backup_dir is required for inplace remap")

    if not inplace:
        if dst_dir is None:
            raise ValueError("dst_dir is required when inplace=False")
        dst_dir.mkdir(parents=True, exist_ok=True)
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    for image_path in sorted(src_dir.iterdir()):
        if not image_path.is_file():
            continue
        label = np.array(Image.open(image_path))
        remapped = remap_label(label, source_value, target_value)

        if inplace:
            shutil.copy2(image_path, backup_dir / image_path.name)
            Image.fromarray(remapped).save(image_path)
        else:
            Image.fromarray(remapped).save(dst_dir / image_path.name)
        file_count += 1
    return file_count


def find_files_with_value(src_dir: Path, target_value: int) -> list[Path]:
    matched = []
    for image_path in sorted(src_dir.iterdir()):
        if image_path.is_file() and np.any(np.array(Image.open(image_path)) == target_value):
            matched.append(image_path)
    return matched


def build_left_frame_path(image_root: Path, filename: str) -> Path:
    stem = Path(filename).stem
    _, seq_id, frame_id = stem.split("_")
    return image_root / f"instrument_dataset_{seq_id}" / "left_frames" / f"{frame_id}.png"


def colorize_label(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for value, color in COLOR_MAP.items():
        rgb[label == value] = color
    return rgb


def export_previews(
    src_dir: Path,
    image_root: Path,
    preview_dir: Path,
    source_value: int,
    target_value: int,
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    matched_files = find_files_with_value(src_dir, target_value=source_value)
    for label_path in matched_files:
        left_frame = build_left_frame_path(image_root, label_path.name)
        image = np.array(Image.open(left_frame).convert("RGB"))
        label_before = np.array(Image.open(label_path))
        label_after = remap_label(label_before, source_value, target_value)

        before_rgb = np.array(Image.fromarray(colorize_label(label_before)).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
        after_rgb = np.array(Image.fromarray(colorize_label(label_after)).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
        panel = np.concatenate([image.astype(np.uint8), before_rgb, after_rgb], axis=1)
        Image.fromarray(panel).save(preview_dir / label_path.name)
    return len(matched_files)


def parse_args():
    parser = argparse.ArgumentParser(description="Task3 label remap utility (copy or inplace).")
    parser.add_argument(
        "--src-dir",
        default="/home/guo/project/data/original_data/endovissub2017-robotics/processing/task3",
        help="Source task3 label directory.",
    )
    parser.add_argument("--source-value", type=int, default=8, help="Label value to replace.")
    parser.add_argument("--target-value", type=int, default=255, help="New label value.")
    parser.add_argument(
        "--mode",
        choices=["copy", "inplace"],
        default="copy",
        help="copy: write into dst-dir. inplace: overwrite src-dir with backup.",
    )
    parser.add_argument(
        "--dst-dir",
        default="/home/guo/project/data/original_data/endovissub2017-robotics/processing_task3_fix8",
        help="Destination directory for copy mode.",
    )
    parser.add_argument(
        "--backup-dir",
        default="/home/guo/project/ssl4mis/data/endovis2017/data/labels_task3_class_backup_before_fix7to0",
        help="Backup directory for inplace mode.",
    )
    parser.add_argument(
        "--preview-dir",
        default="",
        help="Optional preview output dir for files containing source-value (copy mode only).",
    )
    parser.add_argument(
        "--image-root",
        default="/home/guo/project/data/original_data/endovissub2017-robotics/instrument_2017_test",
        help="Root directory containing instrument_dataset_X/left_frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)

    if args.mode == "inplace":
        count = remap_directory(
            src_dir,
            args.source_value,
            args.target_value,
            inplace=True,
            backup_dir=Path(args.backup_dir),
        )
        print(f"Backed up and remapped {count} files in-place: {src_dir}")
        return

    count = remap_directory(
        src_dir,
        args.source_value,
        args.target_value,
        dst_dir=Path(args.dst_dir),
        inplace=False,
    )
    print(f"Remapped {count} files into {args.dst_dir}")

    if args.preview_dir:
        preview_count = export_previews(
            src_dir=src_dir,
            image_root=Path(args.image_root),
            preview_dir=Path(args.preview_dir),
            source_value=args.source_value,
            target_value=args.target_value,
        )
        print(f"Exported {preview_count} preview panels into {args.preview_dir}")


if __name__ == "__main__":
    main()
