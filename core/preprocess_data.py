import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--skip_names", type=str, nargs="*", default=[])
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--resize_h", type=int, default=224)
parser.add_argument("--resize_w", type=int, default=224)
args = parser.parse_args()
input_dir = Path(args.input_dir).resolve()
output_dir = Path(args.output_dir).resolve()
image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
skip_names = [name.lower() for name in args.skip_names]
all_paths = sorted(input_dir.rglob("*"))
files = [src_path for src_path in all_paths if src_path.is_file() and not any(skip_name in str(src_path.relative_to(input_dir)).lower() for skip_name in skip_names)]
for src_path in files:
    (output_dir / src_path.relative_to(input_dir)).parent.mkdir(parents=True, exist_ok=True)
folder_totals = {}
for src_path in files:
    rel_path = src_path.relative_to(input_dir)
    folder_key = str(rel_path.parent)
    folder_totals[folder_key] = folder_totals.get(folder_key, 0) + 1
folder_done = {folder_key: 0 for folder_key in folder_totals}

def process_one(src_path_str, input_dir_str, output_dir_str, skip_names, resize_h, resize_w):
    src_path = Path(src_path_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    rel_path = src_path.relative_to(input_dir)
    dst_path = output_dir / rel_path
    if src_path.suffix.lower() not in image_suffixes:
        if dst_path.exists():
            return str(rel_path), "non-image already exists", None
        shutil.copy2(src_path, dst_path)
        return str(rel_path), "non-image copied as-is", None
    dst_h5_path = dst_path.with_suffix(".h5")
    if dst_h5_path.exists():
        return str(rel_path), "h5 already exists", None
    is_label = "label" in str(rel_path).lower()
    image = Image.open(src_path)
    arr = np.array(image)
    src_shape = arr.shape
    src_min = arr.min().item()
    src_max = arr.max().item()
    if resize_h > 0 and resize_w > 0:
        image = image.resize((resize_w, resize_h), Image.NEAREST if is_label else Image.BILINEAR)
        arr = np.array(image)
    if not is_label:
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    dst_min = arr.min().item()
    dst_max = arr.max().item()
    with h5py.File(dst_h5_path, "w") as f:
        f.create_dataset("img", data=arr, compression=hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE), chunks=arr.shape)
    return str(rel_path), "processed", f"src shape={src_shape} range=({src_min}, {src_max}) -> dst shape={arr.shape} range=({dst_min}, {dst_max})"

if args.num_workers <= 1:
    file_idx = 0
    for src_path in files:
        rel_path, status, detail = process_one(str(src_path), str(input_dir), str(output_dir), skip_names, args.resize_h, args.resize_w)
        file_idx += 1
        folder_key = str(Path(rel_path).parent)
        folder_done[folder_key] += 1
        print(f"[total {file_idx}/{len(files)}] [{folder_key} {folder_done[folder_key]}/{folder_totals[folder_key]}] {rel_path}", flush=True)
        print(f"  {status}", flush=True)
        if detail is not None:
            print(f"  {detail}", flush=True)
else:
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_one, str(src_path), str(input_dir), str(output_dir), skip_names, args.resize_h, args.resize_w) for src_path in files]
        for file_idx, fut in enumerate(as_completed(futures), 1):
            rel_path, status, detail = fut.result()
            folder_key = str(Path(rel_path).parent)
            folder_done[folder_key] += 1
            print(f"[total {file_idx}/{len(files)}] [{folder_key} {folder_done[folder_key]}/{folder_totals[folder_key]}] {rel_path}", flush=True)
            print(f"  {status}", flush=True)
            if detail is not None:
                print(f"  {detail}", flush=True)
