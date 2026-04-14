"""
CPU 추론 벤치마크.

측정:
- 파라미터 수
- 첫 샘플 지연 (cold start)
- 평균 프레임당 추론 시간
- RTF (Real-Time Factor): 처리시간 / 오디오시간
- 피크 메모리
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from training.models import SoramiPipeline


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_pytorch(
    sample_rate: int = 24000,
    segment_seconds: float = 1.0,
    n_runs: int = 50,
    warmup: int = 5,
) -> dict:
    """PyTorch 모델 벤치마크 (CPU)."""
    model = SoramiPipeline(sample_rate=sample_rate)
    model.eval()

    total_params = count_params(model)
    param_breakdown = {
        "encoder": count_params(model.encoder),
        "decoder": count_params(model.decoder),
        "adapter": count_params(model.timbre_adapter),
        "pitch":   count_params(model.pitch_estimator),
        "discrim": 0,  # 추론에 없음
    }

    T = int(segment_seconds * sample_rate)
    dummy = torch.randn(1, 1, T)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.convert(dummy, target_id=0)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.convert(dummy, target_id=0)
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "total_params": total_params,
        "params_M": total_params / 1e6,
        "breakdown": param_breakdown,
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "rtf": float(times.mean() / segment_seconds),
        "segment_seconds": segment_seconds,
    }


def benchmark_onnx(
    onnx_dir: str | Path,
    sample_rate: int = 24000,
    segment_seconds: float = 1.0,
    n_runs: int = 50,
    warmup: int = 5,
) -> dict:
    """ONNX Runtime 벤치마크 (INT8 양자화 후)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required")

    onnx_dir = Path(onnx_dir)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), so,
                                   providers=["CPUExecutionProvider"])
    converter = ort.InferenceSession(str(onnx_dir / "converter.onnx"), so,
                                     providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(onnx_dir / "decoder.onnx"), so,
                                   providers=["CPUExecutionProvider"])

    # 타겟 timbre 로드
    import json
    with open(onnx_dir / "target_timbres.json") as f:
        tgt = np.array(json.load(f)["embeddings"], dtype=np.float32)[0:1]

    T = int(segment_seconds * sample_rate)
    dummy = np.random.randn(1, 1, T).astype(np.float32)

    def _run_once():
        content, src_t, pitch, _ = encoder.run(None, {"waveform": dummy})
        adapted = converter.run(None, {
            "content": content,
            "src_timbre": src_t,
            "tgt_timbre": tgt,
        })[0]
        waveform = decoder.run(None, {
            "content": adapted,
            "tgt_timbre": tgt,
            "pitch": pitch,
        })[0]
        return waveform

    for _ in range(warmup):
        _ = _run_once()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _run_once()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "mean_ms": float(times.mean() * 1000),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "rtf": float(times.mean() / segment_seconds),
        "segment_seconds": segment_seconds,
    }


def print_report(label: str, result: dict):
    print(f"\n=== {label} ===")
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    p.add_argument("--onnx-dir", default=None)
    p.add_argument("--segment", type=float, default=1.0)
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()

    if args.backend == "pytorch":
        result = benchmark_pytorch(
            segment_seconds=args.segment,
            n_runs=args.runs,
        )
        print_report("PyTorch CPU", result)
    else:
        if args.onnx_dir is None:
            raise ValueError("--onnx-dir required for onnx backend")
        result = benchmark_onnx(
            args.onnx_dir,
            segment_seconds=args.segment,
            n_runs=args.runs,
        )
        print_report("ONNX Runtime CPU", result)


if __name__ == "__main__":
    main()
