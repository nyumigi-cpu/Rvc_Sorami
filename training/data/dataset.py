"""
Datasets for Sorami training.

- AudioDataset: 단일 화자 또는 다중 화자 오디오 (self-supervised)
- ParallelAudioDataset: (source, target) 병렬 쌍 (Phase 1 distillation용)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import load_audio, extract_pitch_f0, normalize_loudness, trim_silence
from .augment import RandomAugment


SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _scan_audio_files(root: str | Path) -> list[Path]:
    """디렉토리 재귀 탐색해 오디오 파일 수집."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


class AudioDataset(Dataset):
    """
    단일/다중 화자 오디오 데이터셋.

    각 샘플:
      {
        'waveform':  [1, T],
        'speaker_id': int,
        'pitch':      [T'] (옵션),
        'voicing':    [T'] (옵션),
      }

    매니페스트 형식:
      - 디렉토리: 각 하위 디렉토리 = 화자 (speaker_id는 정렬 순서)
      - JSON: [{"path": ..., "speaker_id": ...}, ...]
    """

    def __init__(
        self,
        root: str | Path | None = None,
        manifest: str | Path | None = None,
        sample_rate: int = 24000,
        segment_seconds: float = 2.0,
        hop_length: int = 240,
        augment: bool = True,
        extract_pitch: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.hop_length = hop_length
        self.extract_pitch = extract_pitch
        self.normalize = normalize

        # 엔트리 로딩
        if manifest is not None:
            with open(manifest, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        elif root is not None:
            entries = []
            root = Path(root)
            # 하위 디렉토리 = 화자
            subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
            if subdirs:
                for sid, d in enumerate(subdirs):
                    for p in _scan_audio_files(d):
                        entries.append({"path": str(p), "speaker_id": sid})
            else:
                # 단일 화자 (모든 파일 speaker_id=0)
                for p in _scan_audio_files(root):
                    entries.append({"path": str(p), "speaker_id": 0})
            self.entries = entries
        else:
            raise ValueError("root 또는 manifest 중 하나는 필수")

        if len(self.entries) == 0:
            raise RuntimeError("No audio files found.")

        self.augment_fn = RandomAugment(sample_rate=sample_rate) if augment else None

    def __len__(self):
        return len(self.entries)

    def _get_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """세그먼트 추출 (랜덤 크롭 또는 우측 패딩)."""
        T = waveform.size(-1)
        if T >= self.segment_samples:
            start = torch.randint(0, T - self.segment_samples + 1, (1,)).item()
            waveform = waveform[..., start : start + self.segment_samples]
        else:
            pad = self.segment_samples - T
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        wav = load_audio(entry["path"], self.sample_rate, mono=True)

        if self.normalize:
            wav = normalize_loudness(wav, self.sample_rate)

        # 세그먼트
        wav = self._get_segment(wav)

        # 피치 추출 (선택)
        sample = {
            "waveform": wav,
            "speaker_id": torch.tensor(entry.get("speaker_id", 0), dtype=torch.long),
        }

        if self.extract_pitch:
            f0, voicing = extract_pitch_f0(
                wav, self.sample_rate, self.hop_length
            )
            sample["pitch"] = torch.from_numpy(f0).float()
            sample["voicing"] = torch.from_numpy(voicing).float()

        # 증강 (피치 추출 후에 적용 — 피치는 원본 기준)
        if self.augment_fn is not None:
            sample["waveform"] = self.augment_fn(sample["waveform"])

        return sample


class ParallelAudioDataset(Dataset):
    """
    병렬 오디오 데이터셋 (Phase 1 증류용).

    Teacher 모델로 생성한 (source_male, target_female) 쌍.

    매니페스트 형식 (JSON):
      [
        {"source": "path_to_male.wav", "target": "path_to_female.wav"},
        ...
      ]
    """

    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int = 24000,
        segment_seconds: float = 2.0,
        augment: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.normalize = normalize

        with open(manifest, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        if len(self.entries) == 0:
            raise RuntimeError(f"Empty manifest: {manifest}")

        self.augment_fn = RandomAugment(sample_rate=sample_rate) if augment else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        src = load_audio(entry["source"], self.sample_rate)
        tgt = load_audio(entry["target"], self.sample_rate)

        if self.normalize:
            src = normalize_loudness(src, self.sample_rate)
            tgt = normalize_loudness(tgt, self.sample_rate)

        # 같은 시작점에서 같은 길이 세그먼트 (정렬되어 있다고 가정)
        min_T = min(src.size(-1), tgt.size(-1))
        if min_T >= self.segment_samples:
            start = torch.randint(0, min_T - self.segment_samples + 1, (1,)).item()
            src = src[..., start : start + self.segment_samples]
            tgt = tgt[..., start : start + self.segment_samples]
        else:
            pad = self.segment_samples - min_T
            src = torch.nn.functional.pad(src[..., :min_T], (0, pad))
            tgt = torch.nn.functional.pad(tgt[..., :min_T], (0, pad))

        if self.augment_fn is not None:
            # 소스에만 증강 (타겟은 깨끗한 reference)
            src = self.augment_fn(src)

        return {
            "source": src,
            "target": tgt,
            "target_id": torch.tensor(entry.get("target_id", 0), dtype=torch.long),
        }


def collate_audio(batch: list[dict]) -> dict:
    """기본 콜레이트 함수 — 고정 길이 세그먼트 가정."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            # 길이 맞춰주기 (최소 길이로 crop)
            if vals[0].dim() >= 1 and any(v.shape != vals[0].shape for v in vals):
                min_len = min(v.size(-1) for v in vals)
                vals = [v[..., :min_len] for v in vals]
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals)
    return out
