"""
Mel-spectrogram reconstruction loss.

Multi-resolution mel loss (HiFi-GAN + UnivNet 스타일):
- 여러 FFT 크기로 mel을 계산해 광대역 + 협대역 정보 모두 활용
- 보코더 학습의 핵심 손실
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelSpectrogram(nn.Module):
    """Log-mel spectrogram 계산. 멀티 해상도 손실의 구성 요소."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        if fmax is None:
            fmax = sample_rate / 2

        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,  # magnitude
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] 또는 [B, 1, T]
        Returns:
            log_mel: [B, n_mels, T']
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        mel = self.mel_scale(waveform)
        # log-mel with small epsilon
        log_mel = torch.log(mel.clamp(min=1e-5))
        return log_mel


class MultiResMelLoss(nn.Module):
    """다중 해상도 mel L1 손실.

    서로 다른 FFT 크기로 mel을 추출해 비교:
    - 큰 FFT: 주파수 해상도 ↑, 시간 해상도 ↓
    - 작은 FFT: 시간 해상도 ↑, 주파수 해상도 ↓
    → 비언어 소리(빠른 전이) + 정상 음성(주파수 정밀도) 모두 잡음
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        resolutions: tuple[tuple[int, int, int], ...] = (
            # (n_fft, hop_length, win_length)
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
        ),
        n_mels: int = 80,
    ):
        super().__init__()
        self.mels = nn.ModuleList([
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                n_mels=n_mels,
            )
            for (n_fft, hop, win) in resolutions
        ])

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_wav:   [B, T] or [B, 1, T]
            target_wav: [B, T] or [B, 1, T] — 같은 길이
        Returns:
            loss: scalar
        """
        # 길이 정렬
        if pred_wav.dim() == 3:
            pred_wav = pred_wav.squeeze(1)
        if target_wav.dim() == 3:
            target_wav = target_wav.squeeze(1)
        min_len = min(pred_wav.size(-1), target_wav.size(-1))
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]

        loss = 0.0
        for mel_fn in self.mels:
            pred_mel = mel_fn(pred_wav)
            target_mel = mel_fn(target_wav)
            # 시간축 길이 정렬
            t = min(pred_mel.size(-1), target_mel.size(-1))
            loss = loss + F.l1_loss(pred_mel[..., :t], target_mel[..., :t])
        return loss / len(self.mels)
