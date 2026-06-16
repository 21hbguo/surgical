import concurrent.futures
import logging
import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from data.transforms import _build_tensor_sample, _normalize_array, _resize_numpy_array, _scale_array_by_dtype
from utils.common import get_task_label_dir


logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _strip_known_image_extension(case):
    root, ext = os.path.splitext(str(case))
    if ext in _IMAGE_EXTENSIONS:
        return root
    return str(case)


def _normalize_label_array(label):
    label = label.astype(np.uint8)
    unique_values = np.unique(label)
    if unique_values.size <= 2 and np.all(np.isin(unique_values, [0, 255])):
        return (label > 0).astype(np.uint8)
    return label


def _load_sample_list(base_dir, split, fold=None, use_val=False):
    if split == "train":
        split_candidates = [f"train_slices_f{fold}.list"] if fold is not None else ["train_slices.list", "train.list"]
    elif split == "val":
        split_candidates = [f"val_slices_f{fold}.list"] if fold is not None else ["val_slices.list", "val.list"]
    else:
        split_candidates = [f"val_slices_f{fold}.list"] if use_val and fold is not None else (["val_slices.list", "val.list"] if use_val else ["test_slices.list", "test.list"])
    list_file_path = None
    for relative_path in split_candidates:
        candidate = os.path.join(base_dir, relative_path)
        if os.path.exists(candidate):
            list_file_path = candidate
            break
    if list_file_path is None:
        if fold is not None and (split != "test" or use_val):
            list_file_path = os.path.join(base_dir, split_candidates[0])
        else:
            return []

    with open(list_file_path, "r", encoding="utf-8") as handle:
        raw_list = [line.strip() for line in handle if line.strip()]
    if split != "test":
        return raw_list
    return [_strip_known_image_extension(item) for item in raw_list]


def _apply_sample_selection(sample_list, split, fold=None, fold_map=None, num=None, sampling="none"):
    filtered = list(sample_list)
    fold_map = fold_map or {}
    if split == "test" and fold is not None and fold in fold_map:
        allowed_seqs = fold_map[fold]
        filtered = [item for item in filtered if int(item.split("_")[1]) in allowed_seqs]

    if num is not None and split in {"train", "val"}:
        num = int(num)
        if split == "train" and sampling == "interval" and 0 < num < len(filtered):
            return [filtered[(i * len(filtered)) // num] for i in range(num)]
        return filtered[:num]
    return filtered


def _find_png_paths(case, base_dir, _split, task):
    case_stem = _strip_known_image_extension(case)
    img_stem = os.path.join(base_dir, "data", "images", case_stem)
    img_path = None
    for ext in _IMAGE_EXTENSIONS:
        candidate = img_stem + ext
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        return None, None

    lab_stem = os.path.join(base_dir, "data", get_task_label_dir(base_dir, task), case_stem)
    for ext in _IMAGE_EXTENSIONS:
        candidate = lab_stem + ext
        if os.path.exists(candidate):
            return img_path, candidate
    return img_path, None


def _find_depth_png_path(case, base_dir, _split, depth_channels=1, depth_uint=16):
    case_stem = _strip_known_image_extension(case)
    path = f"{base_dir}/data/depth{int(depth_channels)}c_slices_uint{int(depth_uint)}/{case_stem}.png"
    return path if os.path.exists(path) else None


def _copy_sample(sample):
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in sample.items()}


def _load_h5_array(path):
    with h5py.File(path, "r") as f:
        return f["img"][()]


def _load_png_sample(case, base_dir, split, resize_size, depth_channels=None, depth_uint=16, task=None):
    target_h, target_w = resize_size
    img_path, lab_path = _find_png_paths(case, base_dir, split, task)
    if img_path is None:
        raise FileNotFoundError(f"Image file not found for case={case!r} under {base_dir}/data/images")
    if lab_path is None:
        raise FileNotFoundError(f"Label file not found for case={case!r} under task-specific labels in {base_dir}/data")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    original_label = _normalize_label_array(lab.copy())
    original_shape = np.array(lab.shape[:2], dtype=np.int64)
    img = _resize_numpy_array(img, (target_h, target_w))
    if split != "test":
        lab = _resize_numpy_array(lab, (target_h, target_w))
    sample = {"image": img, "label": lab, "label_path": lab_path, "original_label": original_label, "original_shape": original_shape}
    if depth_channels == 13:
        depth1_path = _find_depth_png_path(case, base_dir, split, depth_channels=1, depth_uint=depth_uint)
        depth3_path = _find_depth_png_path(case, base_dir, split, depth_channels=3, depth_uint=depth_uint)
        if depth1_path:
            depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
            sample["depth1"] = _resize_numpy_array(depth1, (target_h, target_w))
        if depth3_path:
            depth3 = cv2.imread(depth3_path, cv2.IMREAD_COLOR)
            depth3 = cv2.cvtColor(depth3, cv2.COLOR_BGR2RGB)
            sample["depth3"] = _resize_numpy_array(depth3, (target_h, target_w))
    elif depth_channels == 3:
        depth3_path = _find_depth_png_path(case, base_dir, split, depth_channels=3, depth_uint=depth_uint)
        if depth3_path:
            depth3 = cv2.imread(depth3_path, cv2.IMREAD_COLOR)
            depth3 = cv2.cvtColor(depth3, cv2.COLOR_BGR2RGB)
            sample["depth3"] = _resize_numpy_array(depth3, (target_h, target_w))
    elif depth_channels == 1:
        depth1_path = _find_depth_png_path(case, base_dir, split, depth_channels=1, depth_uint=depth_uint)
        if depth1_path:
            depth1 = cv2.imread(depth1_path, cv2.IMREAD_UNCHANGED)
            sample["depth1"] = _resize_numpy_array(depth1, (target_h, target_w))
    return sample


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        fold=None,
        resize_size=(224, 224),
        load_mode="data",
        depth_channels=None,
        depth_uint=16,
        normalize_method="imagenet",
        sampling="none",
        fold_map=None,
        use_val=False,
        for_inference=False,
        task=None,
    ):
        self._base_dir, self.split, self.transform = base_dir, split, transform
        self.fold = fold
        self.sample_list, self.data_cache, self.max_workers = [], {}, 4
        self.resize_size = resize_size
        self.load_mode = load_mode
        self.depth_channels = depth_channels
        self.depth_uint = int(depth_uint) if depth_uint is not None else 16
        self.normalize_method = normalize_method
        self.sampling = sampling
        self.fold_map = fold_map if fold_map else {}
        self.use_val = use_val
        self.for_inference = for_inference
        self.task = task
        self.sample_list = _load_sample_list(self._base_dir, self.split, fold=self.fold, use_val=self.use_val)
        self.sample_list = _apply_sample_selection(
            self.sample_list,
            self.split,
            fold=self.fold,
            fold_map=self.fold_map,
            num=num,
            sampling=self.sampling,
        )
        if self.split == "test" and self.fold is not None and self.fold in self.fold_map:
            logger.info(
                "Fold %s: filtering to sequences %s, %d samples",
                self.fold,
                self.fold_map[self.fold],
                len(self.sample_list),
            )
        logger.info("total %d samples", len(self.sample_list))
        self._preload_samples()

    def _preload_samples(self):
        if not self.sample_list or self.load_mode != "data":
            return

        logger.info("Multi-core Preloading and Pre-resizing %s data...", self.split)
        tasks = [
            (
                case,
                self._base_dir,
                self.split,
                self.resize_size,
                self.depth_channels,
                self.depth_uint,
                self.task,
            )
            for case in self.sample_list
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_load_png_sample, *task): idx for idx, task in enumerate(tasks)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.sample_list), desc=f"Loading {self.split}"):
                idx = futures[future]
                self.data_cache[idx] = future.result()
        logger.info("Finished preloading %d samples.", len(self.data_cache))

    def __len__(self):
        return len(self.sample_list)

    def _normalize_image_inputs(self, image, depth3=None, depth1=None):
        image = _normalize_array(image, method=self.normalize_method)
        if depth3 is not None:
            depth3 = _scale_array_by_dtype(depth3)
        if depth1 is not None:
            depth1 = _scale_array_by_dtype(depth1)
        return image, depth3, depth1

    def __getitem__(self, idx):
        sample = self._get_sample(idx)
        depth3 = sample.get("depth3") if self.depth_channels else None
        depth1 = sample.get("depth1") if self.depth_channels else None
        image, depth3, depth1 = self._normalize_image_inputs(sample["image"], depth3=depth3, depth1=depth1)
        if self.split == "test" and self.for_inference:
            tensor_sample = _build_tensor_sample(image, sample["label"], depth3=depth3, depth1=depth1)
            tensor_sample["case"] = self.sample_list[idx]
            tensor_sample["original_label"] = sample["original_label"]
            tensor_sample["original_shape"] = np.array(sample.get("original_shape", sample["label"].shape[:2]), dtype=np.int64)
            tensor_sample["original_image"] = sample["image"].copy()
            return tensor_sample
        sample["image"] = image
        if depth3 is not None:
            sample["depth3"] = depth3
        if depth1 is not None:
            sample["depth1"] = depth1
        sample = self.transform(sample) if self.transform else _build_tensor_sample(sample["image"], sample["label"], depth3, depth1)
        sample["idx"] = idx
        return sample

    @staticmethod
    def _find_png_paths_static(case, base_dir, split, task):
        return _find_png_paths(case, base_dir, split, task)

    def _get_sample(self, idx):
        if idx in self.data_cache:
            return _copy_sample(self.data_cache[idx])
        return _load_png_sample(
            self.sample_list[idx],
            self._base_dir,
            self.split,
            self.resize_size,
            self.depth_channels,
            self.depth_uint,
            self.task,
        )


def _load_h5_sample(case, base_dir, task, depth_channels=None, depth_uint=16):
    img_path = os.path.join(base_dir, "data", "images", f"{case}.h5")
    label_dir = get_task_label_dir(base_dir, task)
    lab_path = os.path.join(base_dir, "data", label_dir, f"{case}.h5")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"H5 image not found: {img_path}")
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"H5 label not found: {lab_path}")
    img = _load_h5_array(img_path)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    lab = _load_h5_array(lab_path)
    if lab.dtype != np.uint8:
        lab = lab.astype(np.uint8)
    if lab.ndim == 3 and lab.shape[0] == 1:
        lab = lab[0]
    lab = _normalize_label_array(lab)
    sample = {"image": img, "label": lab, "original_shape": np.array(lab.shape[:2], dtype=np.int64)}
    if depth_channels == 13:
        depth1_path = os.path.join(base_dir, "data", f"depth1c_slices_uint{int(depth_uint)}", f"{case}.h5")
        depth3_path = os.path.join(base_dir, "data", f"depth3c_slices_uint{int(depth_uint)}", f"{case}.h5")
        if os.path.exists(depth1_path):
            depth1 = _load_h5_array(depth1_path)
            sample["depth1"] = depth1 if depth1.dtype == np.float32 else depth1.astype(np.float32)
        if os.path.exists(depth3_path):
            depth3 = _load_h5_array(depth3_path)
            sample["depth3"] = depth3 if depth3.dtype == np.float32 else depth3.astype(np.float32)
    elif depth_channels == 3:
        depth3_path = os.path.join(base_dir, "data", f"depth3c_slices_uint{int(depth_uint)}", f"{case}.h5")
        if os.path.exists(depth3_path):
            depth3 = _load_h5_array(depth3_path)
            sample["depth3"] = depth3 if depth3.dtype == np.float32 else depth3.astype(np.float32)
    elif depth_channels == 1:
        depth1_path = os.path.join(base_dir, "data", f"depth1c_slices_uint{int(depth_uint)}", f"{case}.h5")
        if os.path.exists(depth1_path):
            depth1 = _load_h5_array(depth1_path)
            sample["depth1"] = depth1 if depth1.dtype == np.float32 else depth1.astype(np.float32)
    return sample


class H5DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        fold=None,
        depth_channels=None,
        depth_uint=16,
        sampling="none",
        fold_map=None,
        use_val=False,
        for_inference=False,
        task=None,
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.fold = fold
        self.max_workers = 4
        self.sampling = sampling
        self.depth_channels = depth_channels
        self.depth_uint = int(depth_uint) if depth_uint is not None else 16
        self.fold_map = fold_map or {}
        self.use_val = use_val
        self.for_inference = for_inference
        self.task = task
        self.data_cache = {}
        self.sample_list = _load_sample_list(self._base_dir, self.split, fold=self.fold, use_val=self.use_val)
        self.sample_list = _apply_sample_selection(
            self.sample_list, self.split, fold=self.fold,
            fold_map=self.fold_map, num=num, sampling=self.sampling,
        )
        logger.info("H5DataSets total %d samples, preloading...", len(self.sample_list))
        self._preload_samples()
        logger.info("H5DataSets preloaded %d samples", len(self.data_cache))

    def _preload_samples(self):
        if not self.sample_list:
            return
        logger.info("Multi-core Preloading %s h5 data...", self.split)
        tasks = [
            (
                case,
                self._base_dir,
                self.task,
                self.depth_channels,
                self.depth_uint,
            )
            for case in self.sample_list
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_load_h5_sample, *task): idx for idx, task in enumerate(tasks)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.sample_list), desc=f"Loading {self.split}"):
                idx = futures[future]
                self.data_cache[idx] = future.result()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "test" and self.for_inference:
            sample = _copy_sample(self.data_cache[idx]) if idx in self.data_cache else _load_h5_sample(case, self._base_dir, self.task, self.depth_channels, self.depth_uint)
            tensor_sample = _build_tensor_sample(sample["image"], sample["label"], sample.get("depth3"), sample.get("depth1"))
            tensor_sample["case"] = case
            tensor_sample["original_shape"] = sample.get("original_shape", np.array(sample["label"].shape[:2], dtype=np.int64))
            tensor_sample["original_image"] = sample["image"].copy()
            return tensor_sample
        sample = _copy_sample(self.data_cache[idx]) if idx in self.data_cache else _load_h5_sample(case, self._base_dir, self.task, self.depth_channels, self.depth_uint)
        sample = self.transform(sample) if self.transform else _build_tensor_sample(sample["image"], sample["label"], sample.get("depth3"), sample.get("depth1"))
        sample["idx"] = idx
        return sample
