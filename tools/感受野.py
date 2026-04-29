
# 脚本工具：执行 感受野 相关的数据或分析任务。

import argparse
import importlib
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
import torch
import torch.nn as nn
import torchvision.models as tv_models

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")

RESNET34_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/resnet34-b627a593.pth"
RESNET50_WEIGHTS = "/home/guo/project/ssl4mis/pre_train_ckp/resnet50-11ad3fa6.pth"


# 封装当前模块的主要职责。
@dataclass
class RFState:
    name: str
    layer_type: str
    kernel: int
    stride: int
    padding: int
    dilation: int
    jump: float
    receptive_field: float
    start: float


# 处理2tuple相关逻辑。
def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


# 处理相关内容相关逻辑。
def _layer_params(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        kernel = _to_2tuple(module.kernel_size)[0]
        stride = _to_2tuple(module.stride if module.stride is not None else 1)[0]
        padding = _to_2tuple(module.padding)[0]
        dilation = _to_2tuple(module.dilation)[0] if hasattr(module, "dilation") else 1
        return kernel, stride, padding, dilation
    return None


# 计算相关内容。
def compute_receptive_field(model: nn.Module) -> List[RFState]:
    states: List[RFState] = []
    current_jump = 1.0
    current_rf = 1.0
    current_start = 0.5

    for name, module in model.named_modules():
        if name == "":
            continue
        params = _layer_params(module)
        if params is None:
            continue
        kernel, stride, padding, dilation = params
        effective_kernel = dilation * (kernel - 1) + 1
        next_jump = current_jump * stride
        next_rf = current_rf + (effective_kernel - 1) * current_jump
        next_start = current_start + ((effective_kernel - 1) / 2 - padding) * current_jump
        states.append(
            RFState(
                name=name,
                layer_type=module.__class__.__name__,
                kernel=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                jump=next_jump,
                receptive_field=next_rf,
                start=next_start,
            )
        )
        current_jump = next_jump
        current_rf = next_rf
        current_start = next_start
    return states


# 构建模型。
def build_model(model_name: Optional[str], module_path: Optional[str], factory_name: Optional[str]) -> nn.Module:
    if model_name:
        if model_name == "resnet34":
            model = tv_models.resnet34(weights=None)
            model.load_state_dict(torch.load(RESNET34_WEIGHTS, map_location="cpu"), strict=True)
            return model
        if model_name == "resnet50":
            model = tv_models.resnet50(weights=None)
            model.load_state_dict(torch.load(RESNET50_WEIGHTS, map_location="cpu"), strict=True)
            return model
        factory = getattr(tv_models, model_name)
        try:
            return factory(weights=None)
        except TypeError:
            return factory(pretrained=False)

    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    model = factory()
    return model


# 处理相关内容相关逻辑。
def print_report(states: List[RFState]):
    header = (
        f"{'Layer':<40} {'Type':<12} {'K':>3} {'S':>3} {'P':>3} "
        f"{'D':>3} {'Jump':>8} {'RF':>8} {'Start':>8}"
    )
    print(header)
    print("-" * len(header))
    for state in states:
        print(
            f"{state.name:<40} {state.layer_type:<12} {state.kernel:>3} {state.stride:>3} "
            f"{state.padding:>3} {state.dilation:>3} {state.jump:>8.1f} "
            f"{state.receptive_field:>8.1f} {state.start:>8.1f}"
        )
    if states:
        final = states[-1]
        print("-" * len(header))
        print(
            f"Final receptive field: {final.receptive_field:.1f}, "
            f"effective stride: {final.jump:.1f}, center start: {final.start:.1f}"
        )


# 处理相关内容相关逻辑。
def plot_report(states: List[RFState], save_path: str):
    if not states:
        return
    layer_names = [state.name for state in states]
    rf_values = [state.receptive_field for state in states]
    jump_values = [state.jump for state in states]
    x = list(range(len(states)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(x, rf_values, marker="o", linewidth=2, color="#E74C3C")
    axes[0].set_ylabel("Receptive Field")
    axes[0].set_title("Receptive Field Growth by Layer")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(x, jump_values, marker="o", linewidth=2, color="#3498DB")
    axes[1].set_ylabel("Effective Stride")
    axes[1].set_title("Effective Stride Growth by Layer")
    axes[1].set_xlabel("Layer Index")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    tick_step = max(1, len(layer_names) // 12)
    tick_idx = x[::tick_step]
    tick_labels = [layer_names[i] for i in tick_idx]
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# 解析训练脚本的命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Compute theoretical receptive field for 2D CNN/pooling stacks.")
    parser.add_argument(
        "--model-name",
        default="resnet34",
        help="torchvision model name, for example: resnet34, resnet50, vgg16",
    )
    parser.add_argument("--module-path", default=None, help="Python import path for a custom model factory.")
    parser.add_argument("--factory-name", default=None, help="Callable name returning an nn.Module.")
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional path to save receptive field and effective stride curves.",
    )
    return parser.parse_args()
# 组织脚本主流程。


def main():
    args = parse_args()
    model = build_model(args.model_name, args.module_path, args.factory_name)
    states = compute_receptive_field(model)
    print_report(states)
    if args.plot_path:
        plot_report(states, args.plot_path)
        print(f"Plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
