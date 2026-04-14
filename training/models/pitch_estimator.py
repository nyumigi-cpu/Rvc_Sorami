"""
Pitch Estimator — PESTO 영감의 경량 피치 추정 + 변환

설계 근거:
- PESTO (ISMIR 2023): 130K 파라미터, <10ms 피치 추정
- CREPE의 1/100 크기로 동등 성능
- Self-supervised 학습 가능 (equivariance loss)

남→여 피치 변환 전략:
- 일반적 남성 F0: ~100-150Hz, 여성 F0: ~180-250Hz
- 단순 +octave(×2)는 부자연스러움 (화자 개인차 있음)
- 학습 가능한 비선형 매핑 + 개인 특화 오프셋

파이프라인:
  Audio frame → CNN → Pitch logits (cents grid)
  → Pitch value (Hz)
  → Pitch shift (남→여)
  → Shifted pitch [B, 1, T']

목표: ~130K 파라미터, CPU에서 <1ms 추론
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec_encoder import CausalConv1d


# 피치 범위: MIDI note 기준 (A0=21, A7=105)
# 음성 범위: 대략 60-900Hz → MIDI 34-81
# PESTO는 log-cent grid 사용: C1(32.7Hz)부터 반음 단위로 bin
N_PITCH_BINS = 128  # 128 bins × 50 cents = 64 semitones 커버
FMIN = 32.70  # C1 Hz
CENTS_PER_BIN = 50  # bin당 50 cents (0.5 semitone)


def bins_to_hz(bins: torch.Tensor) -> torch.Tensor:
    """Pitch bin index → Hz."""
    cents = bins * CENTS_PER_BIN
    return FMIN * torch.pow(2.0, cents / 1200.0)


def hz_to_bins(hz: torch.Tensor) -> torch.Tensor:
    """Hz → Pitch bin index (float)."""
    # 0 또는 음수 보호
    hz = hz.clamp(min=1e-5)
    cents = 1200.0 * torch.log2(hz / FMIN)
    return cents / CENTS_PER_BIN


class PitchCNN(nn.Module):
    """경량 CNN 피치 추출기 (PESTO 영감).

    입력: mel-spectrogram 또는 raw CQT 유사 특징 (본 구현에서는 raw audio에서 직접)
    출력: 각 프레임의 pitch probability distribution [B, N_BINS, T']
    """

    def __init__(self, in_channels: int = 1, hidden: int = 32,
                 n_bins: int = N_PITCH_BINS):
        super().__init__()
        # 경량 CNN: 다운샘플링하면서 특징 추출
        # 24kHz 입력 → 100fps 출력 (hop=240)
        self.encoder = nn.Sequential(
            # 24kHz → 8kHz (stride=3)
            CausalConv1d(in_channels, hidden, kernel_size=9, stride=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            # 8kHz → 2kHz (stride=4)
            CausalConv1d(hidden, hidden * 2, kernel_size=9, stride=4),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            # 2kHz → 400fps (stride=5)
            CausalConv1d(hidden * 2, hidden * 2, kernel_size=9, stride=5),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            # 400fps → 100fps (stride=4)
            CausalConv1d(hidden * 2, hidden * 2, kernel_size=9, stride=4),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
        )
        # 문맥 확장 (dilated convs)
        self.context = nn.Sequential(
            CausalConv1d(hidden * 2, hidden * 2, kernel_size=5, dilation=1),
            nn.GELU(),
            CausalConv1d(hidden * 2, hidden * 2, kernel_size=5, dilation=2),
            nn.GELU(),
            CausalConv1d(hidden * 2, hidden * 2, kernel_size=5, dilation=4),
            nn.GELU(),
        )
        # Pitch logits + voicing(유성/무성) 예측
        self.pitch_head = nn.Conv1d(hidden * 2, n_bins, 1)
        self.voicing_head = nn.Conv1d(hidden * 2, 1, 1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: [B, 1, T] — 24kHz raw audio
        Returns:
            pitch_logits: [B, N_BINS, T']
            voicing:      [B, 1, T'] — 유성 확률 (sigmoid 이전)
        """
        x = self.encoder(waveform)
        x = self.context(x)
        pitch_logits = self.pitch_head(x)
        voicing = self.voicing_head(x)
        return pitch_logits, voicing


def decode_pitch(pitch_logits: torch.Tensor,
                 voicing: torch.Tensor,
                 threshold: float = 0.5) -> torch.Tensor:
    """Pitch logits → Hz (soft argmax + voicing masking).

    Args:
        pitch_logits: [B, N_BINS, T']
        voicing:      [B, 1, T'] — sigmoid 이전
        threshold:    유성 판정 임계값

    Returns:
        pitch_hz: [B, 1, T'] — 무성 구간은 0
    """
    # Soft argmax (expected bin)
    probs = F.softmax(pitch_logits, dim=1)  # [B, N, T']
    bins = torch.arange(
        pitch_logits.size(1), device=pitch_logits.device, dtype=pitch_logits.dtype
    ).view(1, -1, 1)
    expected_bin = (probs * bins).sum(dim=1, keepdim=True)  # [B, 1, T']
    pitch_hz = bins_to_hz(expected_bin)

    # Voicing masking
    voicing_prob = torch.sigmoid(voicing)
    mask = (voicing_prob > threshold).float()
    pitch_hz = pitch_hz * mask
    return pitch_hz


class PitchShifter(nn.Module):
    """남→여 피치 변환 — 학습 가능한 비선형 매핑.

    단순히 +octave가 아니라, 화자의 피치 분포에 맞춰 자연스러운 매핑 학습.
    소스 화자(고음/중성적 남성)의 F0 분포 → 타겟 여성 F0 분포.

    매핑 함수: pitch_out = a * pitch_in + b * pitch_in² + c + personal_offset
    (저차원 다항식 + 학습 가능 오프셋)
    """

    def __init__(self, n_targets: int = 1):
        super().__init__()
        # 선형 스케일 (기본값 1.6 ≈ 남→여 피치 비율 중앙값)
        self.linear_scale = nn.Parameter(torch.ones(n_targets) * 1.6)
        # 비선형 보정 (이차항, 작게 시작)
        self.quadratic = nn.Parameter(torch.zeros(n_targets))
        # 상수 오프셋 (Hz)
        self.offset = nn.Parameter(torch.zeros(n_targets))
        # 기준점 (정규화 중심) — 180Hz(남성 평균 근처)
        self.pivot = 180.0

    def forward(self, pitch_hz: torch.Tensor,
                target_id: int | torch.Tensor = 0) -> torch.Tensor:
        """
        Args:
            pitch_hz: [B, 1, T'] — 소스 피치
            target_id: 타겟 화자 ID
        Returns:
            shifted: [B, 1, T'] — 변환된 피치
        """
        if isinstance(target_id, int):
            a = self.linear_scale[target_id]
            q = self.quadratic[target_id]
            b = self.offset[target_id]
        else:
            a = self.linear_scale[target_id].view(-1, 1, 1)
            q = self.quadratic[target_id].view(-1, 1, 1)
            b = self.offset[target_id].view(-1, 1, 1)

        # 무성 구간 (pitch=0)은 보존
        voiced_mask = (pitch_hz > 1.0).float()

        # 정규화된 피치 (pivot 대비)
        normalized = pitch_hz - self.pivot

        # 비선형 매핑
        shifted = a * pitch_hz + q * normalized * normalized / self.pivot + b

        # 피치 양수 보장, 무성은 0
        shifted = shifted.clamp(min=0.0) * voiced_mask
        return shifted


class PitchEstimator(nn.Module):
    """
    통합 피치 추정기.

    두 가지 용도:
    1. 학습 시: 소스와 타겟 오디오에서 F0 추출 → 매핑 학습
    2. 추론 시: 소스 오디오 → F0 → 변환된 F0 (디코더 입력)

    파라미터: ~130K (INT8 ONNX: ~150KB)
    """

    def __init__(self, n_targets: int = 1):
        super().__init__()
        self.cnn = PitchCNN()
        self.shifter = PitchShifter(n_targets=n_targets)

    def estimate(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """피치만 추정 (변환 없이).

        Args:
            waveform: [B, 1, T]
        Returns:
            pitch_hz: [B, 1, T'] — Hz 단위 (무성=0)
            voicing:  [B, 1, T'] — 유성 확률
        """
        pitch_logits, voicing_logits = self.cnn(waveform)
        pitch_hz = decode_pitch(pitch_logits, voicing_logits)
        voicing = torch.sigmoid(voicing_logits)
        return pitch_hz, voicing

    def forward(self, waveform: torch.Tensor,
                target_id: int | torch.Tensor = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """추정 + 변환.

        Args:
            waveform: [B, 1, T]
            target_id: 타겟 화자 ID

        Returns:
            shifted_pitch: [B, 1, T']
            voicing:       [B, 1, T']
        """
        pitch_hz, voicing = self.estimate(waveform)
        shifted = self.shifter(pitch_hz, target_id)
        return shifted, voicing
