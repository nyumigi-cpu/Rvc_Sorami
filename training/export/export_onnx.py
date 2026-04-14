"""
ONNX export for Sorami pipeline.

추론 파이프라인을 3개의 ONNX 모델로 분리:
  1. encoder.onnx  — CodecEncoder (content + timbre 추출)
  2. converter.onnx — TimbreAdapter + PitchShifter (음색/피치 변환)
  3. decoder.onnx  — CodecDecoder (iSTFT 포함)

이렇게 분리하는 이유:
- Rust 추론에서 각 단계 사이에 상태 캐시 관리 가능
- 스트리밍을 위한 배치 크기 동적 변경 가능
- encoder/decoder는 stateless CNN → INT8 양자화 쉬움
"""

from __future__ import annotations

from pathlib import Path
import torch

from ..models import SoramiPipeline


class EncoderWrapper(torch.nn.Module):
    """ONNX 호환 encoder 래퍼."""
    def __init__(self, pipeline: SoramiPipeline):
        super().__init__()
        self.encoder = pipeline.encoder
        self.pitch = pipeline.pitch_estimator if pipeline.use_pitch else None
        self.use_pitch = pipeline.use_pitch

    def forward(self, waveform: torch.Tensor):
        content, src_timbre = self.encoder(waveform)
        if self.use_pitch:
            shifted_pitch, voicing = self.pitch(waveform, target_id=0)
            # content 시간축에 맞추기
            T = content.size(-1)
            if shifted_pitch.size(-1) > T:
                shifted_pitch = shifted_pitch[..., :T]
                voicing = voicing[..., :T]
            elif shifted_pitch.size(-1) < T:
                pad = T - shifted_pitch.size(-1)
                shifted_pitch = torch.nn.functional.pad(shifted_pitch, (0, pad))
                voicing = torch.nn.functional.pad(voicing, (0, pad))
            return content, src_timbre, shifted_pitch, voicing
        else:
            zero = torch.zeros(
                waveform.size(0), 1, content.size(-1),
                device=waveform.device, dtype=waveform.dtype,
            )
            return content, src_timbre, zero, zero


class ConverterWrapper(torch.nn.Module):
    """TimbreAdapter만 ONNX로."""
    def __init__(self, pipeline: SoramiPipeline):
        super().__init__()
        self.adapter = pipeline.timbre_adapter

    def forward(self, content: torch.Tensor, src_timbre: torch.Tensor,
                tgt_timbre: torch.Tensor) -> torch.Tensor:
        return self.adapter(content, src_timbre, tgt_timbre)


class DecoderWrapper(torch.nn.Module):
    """Decoder + iSTFT."""
    def __init__(self, pipeline: SoramiPipeline):
        super().__init__()
        self.decoder = pipeline.decoder

    def forward(self, content: torch.Tensor, tgt_timbre: torch.Tensor,
                pitch: torch.Tensor) -> torch.Tensor:
        return self.decoder(content, tgt_timbre, pitch=pitch)


def _export_module(
    module: torch.nn.Module,
    dummy_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    output_path: str | Path,
    opset: int = 17,
):
    module.eval()
    torch.onnx.export(
        module,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[onnx] Exported: {output_path}")


def export_generator(
    ckpt_path: str | Path,
    output_dir: str | Path,
    segment_seconds: float = 2.0,
    sample_rate: int = 24000,
    opset: int = 17,
):
    """
    학습된 체크포인트 → 3개 ONNX 모델로 내보내기.

    Args:
        ckpt_path: 학습된 체크포인트 경로
        output_dir: ONNX 저장 디렉토리
        segment_seconds: dummy 입력 길이
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ckpt["config"]["model"]
    pipeline = SoramiPipeline(
        content_dim=mcfg["content_dim"],
        timbre_dim=mcfg["timbre_dim"],
        hidden_channels=mcfg["hidden_channels"],
        n_targets=mcfg["n_targets"],
        use_pitch=mcfg["use_pitch"],
        sample_rate=mcfg["sample_rate"],
    )
    pipeline.load_state_dict(ckpt["generator"])
    pipeline.eval()

    # Dummy 입력
    T = int(segment_seconds * sample_rate)
    dummy_wav = torch.randn(1, 1, T)

    # === 1) Encoder ===
    enc_wrapper = EncoderWrapper(pipeline)
    _export_module(
        enc_wrapper,
        (dummy_wav,),
        input_names=["waveform"],
        output_names=["content", "src_timbre", "pitch", "voicing"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "time"},
            "content": {0: "batch", 2: "frames"},
            "src_timbre": {0: "batch"},
            "pitch": {0: "batch", 2: "frames"},
            "voicing": {0: "batch", 2: "frames"},
        },
        output_path=output_dir / "encoder.onnx",
        opset=opset,
    )

    # === 2) Converter ===
    with torch.no_grad():
        content, src_timbre, pitch, _ = enc_wrapper(dummy_wav)
    tgt_timbre = pipeline.target_bank.embeddings[0:1]  # [1, D]
    conv_wrapper = ConverterWrapper(pipeline)
    _export_module(
        conv_wrapper,
        (content, src_timbre, tgt_timbre),
        input_names=["content", "src_timbre", "tgt_timbre"],
        output_names=["adapted_content"],
        dynamic_axes={
            "content": {0: "batch", 2: "frames"},
            "src_timbre": {0: "batch"},
            "tgt_timbre": {0: "batch"},
            "adapted_content": {0: "batch", 2: "frames"},
        },
        output_path=output_dir / "converter.onnx",
        opset=opset,
    )

    # === 3) Decoder ===
    dec_wrapper = DecoderWrapper(pipeline)
    _export_module(
        dec_wrapper,
        (content, tgt_timbre, pitch),
        input_names=["content", "tgt_timbre", "pitch"],
        output_names=["waveform"],
        dynamic_axes={
            "content": {0: "batch", 2: "frames"},
            "tgt_timbre": {0: "batch"},
            "pitch": {0: "batch", 2: "frames"},
            "waveform": {0: "batch", 1: "time"},
        },
        output_path=output_dir / "decoder.onnx",
        opset=opset,
    )

    # 타겟 음색 저장 (JSON)
    import json
    tgt_embed = pipeline.target_bank.embeddings.detach().cpu().numpy().tolist()
    with open(output_dir / "target_timbres.json", "w") as f:
        json.dump({"embeddings": tgt_embed}, f)

    print(f"[onnx] All exports done in {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--segment", type=float, default=2.0)
    args = p.parse_args()
    export_generator(args.ckpt, args.output, segment_seconds=args.segment)
