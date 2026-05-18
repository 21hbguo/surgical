import concurrent.futures
import logging
import os

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from data.transforms import _build_tensor_sample, _normalize_array, _resize_numpy_array
from utils.common import get_task_label_dir

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _strip_known_image_extension(case):
    root, ext = os.path.splitext(str(case))
    if ext in _IMAGE_EXTENSIONS:
        return root
    return str(case)


def _load_sample_list(base_dir, split, fold=None, use_val=False):
    fold_suffix = f"_f{fold}" if fold is not None else ""
    split_map = {
        "train": [f"train_slices{fold_suffix}.list", "train_slices.list", "train.list"],
        "val": [f"val_slices{fold_suffix}.list", "val_slices.list", "val.list"],
        "test": [f"val_slices{fold_suffix}.list", "val_slices.list", "val.list"] if use_val else ["test_slices.list", "test.list"],
    }
    list_file_path = None
    for relative_path in split_map[split]:
        candidate = os.path.join(base_dir, relative_path)
        if os.path.exists(candidate):
            list_file_path = candidate
            break

    if list_file_path is None:
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

    label_dir = get_task_label_dir(base_dir, task)
    lab_stem = os.path.join(base_dir, "data", label_dir, case_stem)
    for ext in _IMAGE_EXTENSIONS:
        candidate = lab_stem + ext
        if os.path.exists(candidate):
            return img_path, candidate
    return img_path, None


def _find_depth_png_path(case, base_dir, _split, depth_channels=1, depth_uint=16):
    case_stem = _strip_known_image_extension(case)
    path = f"{base_dir}/data/depth{int(depth_channels)}c_slices_uint{int(depth_uint)}/{case_stem}.png"
    return path if os.path.exists(path) else None


def _load_and_resize_sample(
    idx,
    case,
    base_dir,
    split,
    target_h,
    target_w,
    num_classes=None,
    depth_channels=None,
    depth_uint=16,
    strategy=None,
    task=None,
):
    del num_classes, strategy

    img_path, lab_path = _find_png_paths(case, base_dir, split, task)
    if img_path is None:
        raise FileNotFoundError(f"Image file not found for case={case!r} under {base_dir}/data/images")
    if lab_path is None:
        raise FileNotFoundError(
            f"Label file not found for case={case!r} under task-specific labels in {base_dir}/data"
        )
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    original_shape = np.array(lab.shape[:2], dtype=np.int64)
    img = _resize_numpy_array(img, (target_h, target_w))
    if split != "test":
        lab = _resize_numpy_array(lab, (target_h, target_w))
    sample = {"image": img, "label": lab, "label_path": lab_path, "original_shape": original_shape}
    if depth_channels is None:
        return idx, sample

    if depth_channels == 13:
        depth1c_path = _find_depth_png_path(case, base_dir, split, depth_channels=1, depth_uint=depth_uint)
        if depth1c_path is not None:
            depth1 = cv2.imread(depth1c_path, cv2.IMREAD_UNCHANGED)
            sample["depth1"] = _resize_numpy_array(depth1, (target_h, target_w))
        depth3c_path = _find_depth_png_path(case, base_dir, split, depth_channels=3, depth_uint=depth_uint)
        if depth3c_path is not None:
            depth3 = cv2.imread(depth3c_path, cv2.IMREAD_COLOR)
            depth3 = cv2.cvtColor(depth3, cv2.COLOR_BGR2RGB)
            sample["depth3"] = _resize_numpy_array(depth3, (target_h, target_w))
        return idx, sample

    if depth_channels == 3:
        depth3c_path = _find_depth_png_path(case, base_dir, split, depth_channels=3, depth_uint=depth_uint)
        if depth3c_path is not None:
            depth3 = cv2.imread(depth3c_path, cv2.IMREAD_COLOR)
            depth3 = cv2.cvtColor(depth3, cv2.COLOR_BGR2RGB)
            sample["depth3"] = _resize_numpy_array(depth3, (target_h, target_w))
        return idx, sample

    depth1c_path = _find_depth_png_path(case, base_dir, split, depth_channels=1, depth_uint=depth_uint)
    if depth1c_path is not None:
        depth1 = cv2.imread(depth1c_path, cv2.IMREAD_UNCHANGED)
        sample["depth1"] = _resize_numpy_array(depth1, (target_h, target_w))
    return idx, sample


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        fold=None,
        max_workers=4,
        resize_size=(224, 224),
        load_mode="data",
        num_classes=None,
        depth_channels=None,
        depth_uint=16,
        strategy=None,
        normalize_method="imagenet",
        sampling="none",
        fold_map=None,
        use_val=False,
        for_inference=False,
        is_depth=None,
        task=None,
    ):
        self._base_dir, self.split, self.transform = base_dir, split, transform
        self.ops_weak, self.ops_strong, self.fold = ops_weak, ops_strong, fold
        self.sample_list, self.data_cache, self.max_workers = [], {}, max_workers
        self.resize_size = resize_size
        self.load_mode = load_mode
        self.num_classes = num_classes
        self.depth_channels = depth_channels
        self.depth_uint = int(depth_uint)
        self.strategy = strategy
        self.normalize_method = normalize_method
        self.sampling = sampling
        self.fold_map = fold_map if fold_map else {}
        self.use_val = use_val
        self.for_inference = for_inference
        self.is_depth = bool(depth_channels) if is_depth is None else bool(is_depth)
        self.task = task
        assert bool(ops_weak) == bool(ops_strong)
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

    @staticmethod
    def _find_png_paths_static(case, base_dir, _split, task):
        return _find_png_paths(case, base_dir, _split, task)

    @staticmethod
    def _find_depth_png_path_static(case, base_dir, _split, depth_channels=1, depth_uint=16):
        return _find_depth_png_path(case, base_dir, _split, depth_channels=depth_channels, depth_uint=depth_uint)

    def _preload_samples(self):
        if not self.sample_list or self.load_mode != "data":
            return

        logger.info("Multi-core Preloading and Pre-resizing %s data...", self.split)
        tasks = [
            (
                idx,
                case,
                self._base_dir,
                self.split,
                self.resize_size[0],
                self.resize_size[1],
                self.num_classes,
                self.depth_channels,
                self.depth_uint,
                self.strategy,
                self.task,
            )
            for idx, case in enumerate(self.sample_list)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_load_and_resize_sample, *task): task[0] for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(self.sample_list), desc=f"Loading {self.split}"):
                idx = futures[future]
                result = future.result()
                if result:
                    self.data_cache[result[0]] = result[1]
        logger.info("Finished preloading %d samples.", len(self.data_cache))

    def __len__(self):
        return len(self.sample_list)

    def _get_sample(self, idx):
        if idx in self.data_cache:
            return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self.data_cache[idx].items()}
        return _load_and_resize_sample(
            idx,
            self.sample_list[idx],
            self._base_dir,
            self.split,
            *self.resize_size,
            self.num_classes,
            self.depth_channels,
            self.depth_uint,
            self.strategy,
            self.task,
        )[1]

    def _normalize_inputs(self, image, depth3=None, depth1=None):
        image = _normalize_array(image, method=self.normalize_method)
        if depth3 is not None:
            depth3 = _normalize_array(depth3, method=self.normalize_method)
        if depth1 is not None:
            if self.normalize_method == "255":
                depth1_dtype = depth1.dtype
                depth1 = depth1.astype(np.float32)
                if np.issubdtype(depth1_dtype, np.integer):
                    depth1_max = float(np.iinfo(depth1_dtype).max)
                    if depth1_max > 1.0 and float(depth1.max()) > 1.0:
                        depth1 = depth1 / depth1_max
                elif float(depth1.max()) > 1.0:
                    depth1 = depth1 / 255.0
            else:
                depth1 = _normalize_array(depth1, method=self.normalize_method)
        return image, depth3, depth1

    def _to_train_or_val_item(self, sample, idx):
        image, depth3, depth1 = self._normalize_inputs(sample["image"], depth3=sample.get("depth3"), depth1=sample.get("depth1"))
        sample["image"] = image
        if depth3 is not None:
            sample["depth3"] = depth3
        if depth1 is not None:
            sample["depth1"] = depth1
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = _build_tensor_sample(sample["image"], sample["label"], depth3, depth1)
        sample["idx"] = idx
        return sample

    def _to_test_inference_item(self, sample, idx):
        depth3 = sample.get("depth3") if self.is_depth else None
        depth1 = sample.get("depth1") if self.is_depth else None
        image, depth3, depth1 = self._normalize_inputs(sample["image"], depth3=depth3, depth1=depth1)
        tensor_sample = _build_tensor_sample(image, sample["label"], depth3=depth3, depth1=depth1)
        original_shape = sample.get("original_shape")
        if original_shape is None:
            original_shape = np.array(sample["label"].shape[:2], dtype=np.int64)
        tensor_sample["case"] = self.sample_list[idx]
        tensor_sample["original_shape"] = np.array(original_shape)
        tensor_sample["original_image"] = sample["image"].copy()
        return tensor_sample

    def __getitem__(self, idx):
        sample = self._get_sample(idx)

        if self.split in {"train", "val"}:
            return self._to_train_or_val_item(sample, idx)

        if self.split == "test" and self.for_inference:
            return self._to_test_inference_item(sample, idx)

        if self.split == "test":
            return self._to_train_or_val_item(sample, idx)

        raise ValueError(f"Unsupported split: {self.split}")


def _load_h5_sample(case, base_dir, task, target_h, target_w):
    img_path = os.path.join(base_dir, "data", "images", f"{case}.h5")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"H5 image not found: {img_path}")

    label_dir = get_task_label_dir(base_dir, task)
    lab_path = os.path.join(base_dir, "data", label_dir, f"{case}.h5")
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"H5 label not found: {lab_path}")

    with h5py.File(img_path, "r") as f:
        img = f["image"][()].astype(np.float32)
    with h5py.File(lab_path, "r") as f:
        lab = f["label"][()].astype(np.uint8)

    original_shape = np.array(lab.shape[:2], dtype=np.int64)
    img = _resize_numpy_array(img, (target_h, target_w))
    lab = _resize_numpy_array(lab, (target_h, target_w))
    return {"image": img, "label": lab, "original_shape": original_shape}


class H5DataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
        fold=None,
        max_workers=4,
        resize_size=(224, 224),
        num_classes=None,
        normalize_method="imagenet",
        sampling="none",
        fold_map=None,
        use_val=False,
        for_inference=False,
        task=None,
    ):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.fold = fold
        self.resize_size = resize_size
        self.num_classes = num_classes
        self.normalize_method = normalize_method
        self.sampling = sampling
        self.fold_map = fold_map or {}
        self.use_val = use_val
        self.for_inference = for_inference
        self.task = task
        self.data_cache = {}

        assert bool(ops_weak) == bool(ops_strong)
        self.sample_list = _load_sample_list(self._base_dir, self.split, fold=self.fold, use_val=self.use_val)
        self.sample_list = _apply_sample_selection(
            self.sample_list, self.split, fold=self.fold,
            fold_map=self.fold_map, num=num, sampling=self.sampling,
        )
        logger.info("H5DataSets total %d samples, preloading...", len(self.sample_list))
        self._preload_samples()
        logger.info("H5DataSets preloaded %d samples", len(self.data_cache))

    def _preload_samples(self):
        for idx, case in enumerate(self.sample_list):
            self.data_cache[idx] = _load_h5_sample(
                case, self._base_dir, self.task,
                self.resize_size[0], self.resize_size[1],
            )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if idx in self.data_cache:
            sample = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.data_cache[idx].items()}
        else:
            sample = _load_h5_sample(
                case, self._base_dir, self.task,
                self.resize_size[0], self.resize_size[1],
            )

        if self.split in {"train", "val"}:
            image, _, _ = self._normalize_inputs(sample["image"])
            sample["image"] = image
            if self.transform:
                sample = self.transform(sample)
            else:
                sample = _build_tensor_sample(sample["image"], sample["label"])
            sample["idx"] = idx
            return sample

        if self.split == "test" and self.for_inference:
            image, _, _ = self._normalize_inputs(sample["image"])
            sample["image"] = image
            tensor_sample = _build_tensor_sample(image, sample["label"])
            tensor_sample["case"] = case
            tensor_sample["original_shape"] = sample.get("original_shape", np.array(sample["label"].shape[:2], dtype=np.int64))
            tensor_sample["original_image"] = sample["image"].copy()
            return tensor_sample

        if self.split == "test":
            image, _, _ = self._normalize_inputs(sample["image"])
            sample["image"] = image
            sample = _build_tensor_sample(sample["image"], sample["label"])
            sample["idx"] = idx
            return sample

        raise ValueError(f"Unsupported split: {self.split}")

    def _normalize_inputs(self, image, depth3=None, depth1=None):
        image = _normalize_array(image, method=self.normalize_method)
        if depth3 is not None:
            depth3 = _normalize_array(depth3, method=self.normalize_method)
        if depth1 is not None:
            depth1 = _normalize_array(depth1, method=self.normalize_method)
        return image, depth3, depth1
