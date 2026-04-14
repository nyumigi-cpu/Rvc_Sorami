"""
오디오 변환 테스트.

입력 wav → 변환된 wav 저장.
학습 체크포인트가 필요하지만, 없으면 무작위 초기화된 모델로 스모크 테스트.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from training.models import SoramiPipeline


def convert_file(
    input_path: str,
    output_path: str,
    ckpt_path: str | None = None,
    target_id: int = 0,
    device: str = "cpu",
):
    # 모델 로드
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        mcfg = ckpt["config"]["model"]
        model = SoramiPipeline(
            content_dim=mcfg["content_dim"],
            timbre_dim=mcfg["timbre_dim"],
            hidden_channels=mcfg["hidden_channels"],
            n_targets=mcfg["n_targets"],
            use_pitch=mcfg["use_pitch"],
            sample_rate=mcfg["sample_rate"],
        )
        model.load_state_dict(ckpt["generator"])
        print(f"[load] {ckpt_path}")
    else:
        print("[load] No checkpoint — using random initialization (smoke test)")
        model = SoramiPipeline()

    model.to(device)
    model.eval()
    sr = model.sample_rate

    # 오디오 로드
    waveform, orig_sr = torchaudio.load(input_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    # 변환
    with torch.no_grad():
        converted = model.convert(waveform, target_id=target_id)

    if converted.dim() == 2:
        converted = converted.unsqueeze(0)

    # 정규화 후 저장
    peak = converted.abs().max()
    if peak > 0.99:
        converted = converted * (0.99 / peak)

    torchaudio.save(output_path, converted.squeeze(0).cpu(), sr)
    print(f"[save] {output_path}  ({converted.size(-1) / sr:.2f}s @ {sr}Hz)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--target-id", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    convert_file(args.input, args.output, args.ckpt, args.target_id, args.device)


if __name__ == "__main__":
    main()
