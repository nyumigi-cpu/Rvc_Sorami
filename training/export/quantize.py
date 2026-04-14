"""
ONNX INT8 동적 양자화.

CPU 추론에서 크기 75% 감소, 속도 1.5-2x 향상.
동적 양자화는 보정 데이터 없이도 작동 (정적 양자화는 보정 필요).
"""

from __future__ import annotations

from pathlib import Path


def quantize_int8(
    input_onnx: str | Path,
    output_onnx: str | Path,
    per_channel: bool = False,
):
    """ONNX 모델에 INT8 동적 양자화 적용.

    Args:
        input_onnx:  원본 FP32 ONNX 경로
        output_onnx: 양자화된 INT8 ONNX 저장 경로
        per_channel: True면 채널별 양자화 (품질↑, 속도↓)
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime required: pip install onnxruntime onnxruntime-tools"
        )

    quantize_dynamic(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    print(f"[quantize] {input_onnx} → {output_onnx} (INT8)")


def quantize_all(onnx_dir: str | Path):
    """디렉토리 내 모든 ONNX를 INT8로 양자화."""
    onnx_dir = Path(onnx_dir)
    for onnx_path in onnx_dir.glob("*.onnx"):
        if onnx_path.stem.endswith(".int8"):
            continue  # 이미 양자화됨
        out = onnx_path.with_name(onnx_path.stem + ".int8.onnx")
        quantize_int8(onnx_path, out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="ONNX 파일 또는 디렉토리")
    p.add_argument("--output", default=None, help="출력 경로 (디렉토리면 자동 설정)")
    args = p.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        quantize_all(input_path)
    else:
        output = args.output or str(input_path.with_name(
            input_path.stem + ".int8.onnx"
        ))
        quantize_int8(input_path, output)
