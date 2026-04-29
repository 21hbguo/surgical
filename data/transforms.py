import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom


def _normalize_array(array: np.ndarray, method: str = "minmax") -> np.ndarray:
    array = array.astype(np.float32)
    if method == "minmax":
        a_min, a_max = array.min(), array.max()
        if a_max > a_min:
            return (array - a_min) / (a_max - a_min)
        return np.zeros_like(array)
    if method == "255":
        if array.max() > 1.0:
            return array / 255.0
        return array
    if method == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if array.max() > 1.0:
            array = array / 255.0
        if array.ndim == 3 and array.shape[2] == 3:
            for c in range(3):
                array[..., c] = (array[..., c] - mean[c]) / std[c]
        elif array.ndim == 3 and array.shape[2] == 1:
            array[..., 0] = (array[..., 0] - mean[0]) / std[0]
        elif array.ndim == 2:
            array = (array - mean[0]) / std[0]
        return array
    return array


def _resize_numpy_array(v: np.ndarray, target_size: tuple) -> np.ndarray:
    th, tw = target_size[-2:]
    if v.ndim == 2:
        h, w = v.shape
        if (h, w) == (th, tw):
            return v
        return zoom(v, (th / h, tw / w), order=0, mode="reflect")
    if v.ndim == 3:
        if v.shape[-1] == 3:
            h, w = v.shape[:2]
            if (h, w) == (th, tw):
                return v
            return zoom(v, (th / h, tw / w, 1), order=0, mode="reflect")
        c, h, w = v.shape
        if (h, w) == (th, tw):
            return v
        return np.stack(
            [zoom(v[i], (th / h, tw / w), order=0, mode="reflect") for i in range(c)],
            axis=0,
        )
    return v


def array_to_tensor(array, is_label=False):
    if is_label:
        label = array.astype(np.uint8)
        unique_values = np.unique(label)
        if unique_values.size <= 2 and np.all(np.isin(unique_values, [0, 255])):
            label = (label > 0).astype(np.uint8)
        return torch.from_numpy(label).long()
    array = array.astype(np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        return tensor.permute(2, 0, 1)
    return tensor


def _build_tensor_sample(image, label, depth3=None, depth1=None, image_s=None):
    sample = {
        "image": array_to_tensor(image),
        "label": array_to_tensor(label, is_label=True),
    }
    if image_s is not None:
        sample["image_s"] = image_s
    if depth3 is not None:
        sample["depth3"] = array_to_tensor(depth3)
    if depth1 is not None:
        sample["depth1"] = array_to_tensor(depth1)
    return sample


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        return np.repeat(depth[..., None], 3, axis=2)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return np.repeat(depth, 3, axis=2)
    if depth.ndim == 3 and depth.shape[-1] == 3:
        return depth
    return depth


class RandomGenerator(object):
    def __init__(self, resize_size=None, is_val=False, root_path=None, depth_channels=None):
        self.resize_size, self.is_val = resize_size, is_val
        self.root_path, self.depth_channels = root_path, depth_channels

    def __call__(self, sample):
        img, lab = sample["image"], sample["label"]
        depth3 = sample.get("depth3")
        depth1 = sample.get("depth1")
        if self.is_val:
            return _build_tensor_sample(img, lab, depth3, depth1)

        if random.random() < 0.5:
            img, lab, depth3, depth1 = self._random_rot_flip(img, lab, depth3, depth1)
        else:
            img, lab, depth3, depth1 = self._random_rotate(img, lab, depth3, depth1)

        if self.resize_size is not None and img.shape[:2] != tuple(self.resize_size):
            img = _resize_numpy_array(img, self.resize_size)
            lab = _resize_numpy_array(lab, self.resize_size)
            if depth3 is not None:
                depth3 = _resize_numpy_array(depth3, self.resize_size)
            if depth1 is not None:
                depth1 = _resize_numpy_array(depth1, self.resize_size)
        return _build_tensor_sample(img, lab, depth3, depth1)

    def _random_rot_flip(self, img, lab, depth3, depth1):
        k = np.random.randint(0, 4)
        img = np.rot90(img, k)
        lab = np.rot90(lab, k)
        axis = np.random.randint(0, 2)
        img = np.flip(img, axis=axis).copy()
        lab = np.flip(lab, axis=axis).copy()
        if depth3 is not None:
            depth3 = np.rot90(depth3, k)
            depth3 = np.flip(depth3, axis=axis).copy()
        if depth1 is not None:
            depth1 = np.rot90(depth1, k)
            depth1 = np.flip(depth1, axis=axis).copy()
        return img, lab, depth3, depth1

    def _random_rotate(self, img, lab, depth3, depth1):
        angle = np.random.randint(-20, 20)
        img = ndimage.rotate(img, angle, order=0, reshape=False, mode="reflect")
        lab = ndimage.rotate(lab, angle, order=0, reshape=False, mode="reflect")
        if depth3 is not None:
            depth3 = ndimage.rotate(depth3, angle, order=0, reshape=False, mode="reflect")
        if depth1 is not None:
            depth1 = ndimage.rotate(depth1, angle, order=0, reshape=False, mode="reflect")
        return img, lab, depth3, depth1
