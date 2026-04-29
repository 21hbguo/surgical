import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from tools.域解释 import (
        BATCH_SIZE as DEFAULT_BATCH_SIZE,
        DinoV2Extractor,
        DinoV3Extractor,
        OUTPUT_DIR,
        RGBExtractor,
        RGBExtractor34,
        SEED,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from 域解释 import (
        BATCH_SIZE as DEFAULT_BATCH_SIZE,
        DinoV2Extractor,
        DinoV3Extractor,
        OUTPUT_DIR,
        RGBExtractor,
        RGBExtractor34,
        SEED,
    )

IMAGE_SIZE = 224
MODEL_BUILDERS = {
    "resnet50": RGBExtractor,
    "resnet34": RGBExtractor34,
    "dinov2_vits14": DinoV2Extractor,
    "dinov3_vits16": DinoV3Extractor,
}


def build_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)


def benchmark_inference(model_name: str, model: torch.nn.Module, device: torch.device, batch_size: int, warmup_steps: int, test_steps: int) -> dict:
    batch = build_batch(batch_size, device)
    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        last_output = None
        for _ in range(test_steps):
            last_output = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "device": str(device),
        "batch_shape": list(batch.shape),
        "output_shape": list(last_output.shape) if last_output is not None else [],
        "warmup_steps": warmup_steps,
        "test_steps": test_steps,
        "avg_forward_ms": elapsed / test_steps * 1000.0,
        "images_per_sec": batch_size * test_steps / elapsed,
    }


def benchmark_train_step(model_name: str, model: torch.nn.Module, device: torch.device, batch_size: int, warmup_steps: int, test_steps: int) -> dict:
    model = model.to(device).train()
    batch = build_batch(batch_size, device)

    with torch.no_grad():
        probe = model(batch[:1]).detach()
    out_dim = probe.shape[-1]
    target = torch.randn(batch_size, out_dim, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    last_output = None
    last_loss = None
    for _ in range(test_steps):
        optimizer.zero_grad(set_to_none=True)
        last_output = model(batch)
        last_loss = torch.nn.functional.mse_loss(last_output, target)
        last_loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "device": str(device),
        "batch_shape": list(batch.shape),
        "output_shape": list(last_output.shape),
        "warmup_steps": warmup_steps,
        "test_steps": test_steps,
        "avg_train_step_ms": elapsed / test_steps * 1000.0,
        "images_per_sec": batch_size * test_steps / elapsed,
        "final_loss": float(last_loss.item()),
        "max_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024 / 1024) if device.type == "cuda" else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Domain extractor benchmark (inference/train).")
    parser.add_argument("--mode", choices=["infer", "infer-multibatch", "train"], default="infer")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--test-steps", type=int, default=50)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    if args.mode == "infer-multibatch":
        for model_name, builder in MODEL_BUILDERS.items():
            for batch_size in args.batch_sizes:
                model = builder()
                results.append(
                    benchmark_inference(
                        model_name=model_name,
                        model=model,
                        device=device,
                        batch_size=batch_size,
                        warmup_steps=args.warmup_steps,
                        test_steps=args.test_steps,
                    )
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    elif args.mode == "train":
        for model_name, builder in MODEL_BUILDERS.items():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = builder()
            results.append(
                benchmark_train_step(
                    model_name=model_name,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                    warmup_steps=max(1, args.warmup_steps // 2),
                    test_steps=max(1, args.test_steps // 2),
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        for model_name, builder in MODEL_BUILDERS.items():
            model = builder()
            results.append(
                benchmark_inference(
                    model_name=model_name,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    test_steps=args.test_steps,
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_name = {
        "infer": "domain_model_speed_benchmark.json",
        "infer-multibatch": "domain_model_speed_benchmark_multibatch.json",
        "train": "domain_model_train_benchmark.json",
    }[args.mode]
    output_path = Path(args.output) if args.output else output_dir / default_name
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved benchmark to: {output_path}")


if __name__ == "__main__":
    main()
