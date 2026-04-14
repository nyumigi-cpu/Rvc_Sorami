"""
Audio preprocessing utilities.

- load_audio: 리샘플링 포함 로드
- extract_pitch_f0: pyworld DIO/Harvest로 F0 추출 (학습용 타겟)
- normalize_loudness: LUFS 기반 loudness 정규화
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio


def load_audio(path: str, sample_rate: int = 24000, mono: bool = True) -> torch.Tensor:
    """오디오 파일 로드 + 리샘플링.

    Returns:
        waveform: [1, T] (mono) or [C, T]
    """
    waveform, sr = torchaudio.load(path)
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def extract_pitch_f0(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int = 24000,
    hop_length: int = 240,
    f0_min: float = 50.0,
    f0_max: float = 800.0,
    method: str = "harvest",
) -> tuple[np.ndarray, np.ndarray]:
    """F0 추출 (pyworld 기반).

    Args:
        waveform: [T] 또는 [1, T]
        method: 'harvest' (정확) 또는 'dio' (빠름)
    Returns:
        f0:      [T_frames] — Hz, 무성=0
        voicing: [T_frames] — 0 또는 1
    """
    try:
        import pyworld as pw
    except ImportError:
        raise ImportError("pyworld required: pip install pyworld")

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().cpu().numpy()
    waveform = waveform.astype(np.float64)

    frame_period = 1000.0 * hop_length / sample_rate  # ms

    if method == "harvest":
        f0, t = pw.harvest(
            waveform, sample_rate,
            f0_floor=f0_min, f0_ceil=f0_max,
            frame_period=frame_period,
        )
    else:
        f0, t = pw.dio(
            waveform, sample_rate,
            f0_floor=f0_min, f0_ceil=f0_max,
            frame_period=frame_period,
        )
        f0 = pw.stonemask(waveform, f0, t, sample_rate)

    voicing = (f0 > 0).astype(np.float32)
    return f0.astype(np.float32), voicing


def normalize_loudness(
    waveform: torch.Tensor,
    sample_rate: int = 24000,
    target_lufs: float = -23.0,
) -> torch.Tensor:
    """LUFS 기반 loudness 정규화. pyloudnorm 필요.

    폴백: peak normalization (|x| ≤ 0.95).
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        # 폴백: 피크 정규화
        peak = waveform.abs().max()
        if peak > 1e-5:
            waveform = waveform * (0.95 / peak)
        return waveform

    wav_np = waveform.squeeze().cpu().numpy()
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(wav_np)
        normalized = pyln.normalize.loudness(wav_np, loudness, target_lufs)
    except Exception:
        # 너무 짧거나 무음 → 피크 정규화
        peak = np.abs(wav_np).max()
        normalized = wav_np * (0.95 / (peak + 1e-8))

    return torch.from_numpy(normalized).unsqueeze(0).float()


def trim_silence(
    waveform: torch.Tensor,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """양 끝의 무음 제거 (librosa.effects.trim과 유사).

    폴백: librosa 없으면 에너지 기반 간단 구현.
    """
    try:
        import librosa
        wav_np = waveform.squeeze().cpu().numpy()
        trimmed, _ = librosa.effects.trim(
            wav_np,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return torch.from_numpy(trimmed).unsqueeze(0).float()
    except ImportError:
        pass

    # 폴백
    abs_wav = waveform.abs().squeeze()
    threshold = 10 ** (threshold_db / 20.0)
    above = abs_wav > threshold
    if above.any():
        nz = above.nonzero(as_tuple=True)[0]
        start, end = nz.min().item(), nz.max().item() + 1
        return waveform[..., start:end]
    return waveform
