"""
Discriminator — Multi-Period + Multi-Scale Discriminator

설계 근거:
- HiFi-GAN의 MPD + MSD 조합이 음성 합성 GAN의 표준
- MPD: 주기적 패턴 감지 (모음/음정 구조)
- MSD: 다중 스케일 시간 구조 (음소 전이, 비언어 소리)
- 판별기는 학습용이므로 크기에 제약 없음 (ONNX 변환 안 함)

학습에만 사용 → 추론 모델 크기에 영향 없음.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1


class PeriodDiscriminator(nn.Module):
    """한 주기 p에 대한 판별기. HiFi-GAN 오리지널."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                           padding=(kernel_size // 2, 0))),
            norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                           padding=(kernel_size // 2, 0))),
            norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                           padding=(kernel_size // 2, 0))),
            norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                           padding=(kernel_size // 2, 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,
                           padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: [B, 1, T] 또는 [B, T]
        Returns:
            logit: [B, 1, T_out, 1]
            feature_maps: list — feature matching loss용
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        fmaps = []
        b, c, t = x.shape
        # 주기 p로 패딩 후 reshape: [B, C, T/p, p]
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    """MPD — 서로 다른 주기를 가진 여러 판별기.

    주기: 2, 3, 5, 7, 11 (서로소 소수)
    각 주기로 파형을 2D로 reshape해 패턴 감지.
    """

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        """
        Returns:
            logits: list of [B, *]
            fmaps:  list of list (각 sub-discriminator별 feature maps)
        """
        logits, fmaps = [], []
        for d in self.discriminators:
            logit, fmap = d(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class ScaleDiscriminator(nn.Module):
    """한 스케일에 대한 시간 도메인 판별기."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fmaps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    """MSD — 원본/2x 다운샘플/4x 다운샘플 세 스케일에서 판별.

    스케일: 원본(24kHz), 12kHz, 6kHz
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),   # 원본
            ScaleDiscriminator(use_spectral_norm=False),  # 2x 다운
            ScaleDiscriminator(use_spectral_norm=False),  # 4x 다운
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        logits, fmaps = [], []
        for i, (d, pool) in enumerate(zip(self.discriminators, self.pools)):
            x = pool(x) if i > 0 else x
            logit, fmap = d(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class CombinedDiscriminator(nn.Module):
    """MPD + MSD 통합. 학습 루프에서 단일 인터페이스로 사용."""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x: torch.Tensor) -> tuple[list, list]:
        """
        Args:
            x: [B, 1, T] 또는 [B, T]
        Returns:
            logits: MPD logits + MSD logits
            fmaps:  MPD fmaps  + MSD fmaps
        """
        mpd_logits, mpd_fmaps = self.mpd(x)
        msd_logits, msd_fmaps = self.msd(x)
        return mpd_logits + msd_logits, mpd_fmaps + msd_fmaps
