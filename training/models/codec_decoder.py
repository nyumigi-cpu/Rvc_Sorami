"""
Causal Codec Decoder — 경량 인과적 오디오 디코더

설계 근거:
- Vocos 방식: iSTFT 헤드로 보코더 연산 대폭 감소 (CPU 169.63x 실시간)
- MS-Wavehax 영감: 극경량 CNN 기반 보코더 (0.332M 파라미터)
- QuickVC 방식: iSTFT 디코더로 시간 도메인 직접 생성 대신 주파수 도메인 예측
- StreamVC 방식: 인과적 업샘플링

파이프라인:
  Content tokens + Timbre (변환된) → Causal Upsample Blocks
  → ConvNeXt-style Refinement → STFT coefficients (magnitude + phase)
  → iSTFT → Raw waveform (24kHz)

목표: ~1.5M 파라미터, CPU에서 5-8ms 추론
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec_encoder import CausalConv1d


class CausalConvTranspose1d(nn.Module):
    """인과적 전치 합성곱 — 업샘플링 시 미래 정보를 참조하지 않음.

    일반 ConvTranspose1d는 양쪽으로 출력이 번지지만,
    인과성을 유지하려면 출력의 오른쪽(미래) 부분을 잘라낸다.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        # 오른쪽에서 잘라낼 샘플 수 (인과성 유지)
        self.trim = kernel_size - stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.trim > 0:
            x = x[..., : -self.trim]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 블록 — Vocos에서 사용, LayerNorm 대신 BN 사용 (스트리밍).

    구조: DWConv → BN → PWConv(expand) → GELU → PWConv(contract) → residual
    """

    def __init__(self, channels: int, kernel_size: int = 7, expand_ratio: int = 3):
        super().__init__()
        hidden = channels * expand_ratio
        self.dwconv = CausalConv1d(channels, channels, kernel_size, groups=channels)
        self.norm = nn.BatchNorm1d(channels)
        self.pwconv1 = nn.Conv1d(channels, hidden, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden, channels, 1)
        # LayerScale: 초기값 작게 설정해 학습 안정화
        self.gamma = nn.Parameter(torch.ones(channels, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return x + residual


class UpsampleBlock(nn.Module):
    """업샘플 블록 — Causal Transpose Conv + ConvNeXt refinement."""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 n_refine: int = 2):
        super().__init__()
        # 업샘플 (커널 크기 = stride * 2로 겹침 효과)
        self.upsample = CausalConvTranspose1d(
            in_channels, out_channels,
            kernel_size=stride * 2, stride=stride,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.refine = nn.ModuleList([
            ConvNeXtBlock(out_channels) for _ in range(n_refine)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm(x)
        x = self.act(x)
        for block in self.refine:
            x = block(x)
        return x


class ISTFTHead(nn.Module):
    """iSTFT 헤드 — magnitude + phase를 직접 예측해 iSTFT로 파형 합성.

    Vocos 방식:
    - 시간 도메인 파형을 직접 생성하는 대신, STFT 계수를 예측
    - iSTFT는 미분 가능한 결정론적 연산이라 학습 가능
    - 연산량이 크게 감소 (CPU 친화적)
    """

    def __init__(self, in_channels: int, n_fft: int = 480, hop_length: int = 240):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        # 출력: (magnitude, phase) × (n_fft/2 + 1) freq bins
        n_bins = n_fft // 2 + 1
        self.proj = nn.Conv1d(in_channels, n_bins * 2, 1)
        # 윈도우 (학습 중에는 고정)
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T_frames] — 프레임 단위 특징
        Returns:
            waveform: [B, T_samples] — 시간 도메인 파형
        """
        x = self.proj(x)  # [B, 2*n_bins, T_frames]
        mag, phase = x.chunk(2, dim=1)

        # magnitude는 양수여야 함 (exp로 매핑)
        # clamp로 폭주 방지 (log magnitude ≤ ~9 → mag ≤ ~8000)
        mag = torch.exp(mag.clamp(max=9.0))
        # phase는 [-π, π] 범위로 매핑 (cos/sin 사용)
        # 네트워크가 phase 각도 자체를 예측 → 주기적 활성화로 안정화
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        # Complex spectrogram: [B, n_bins, T_frames]
        real = mag * cos_p
        imag = mag * sin_p
        spec = torch.complex(real, imag)

        # iSTFT (center=True여도 인과성 문제 없음: 학습 시 과거만 이용해 mag/phase 예측함)
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            return_complex=False,
        )
        return waveform


class CodecDecoder(nn.Module):
    """
    경량 인과적 코덱 디코더.

    입력:
      - content_features [B, content_dim, T'] — 콘텐츠/구조
      - timbre_embedding [B, timbre_dim] — 타겟 화자 음색 (이미 변환된 것)
      - pitch [B, 1, T']  — F0 곡선 (옵션, 변환된 피치)

    출력:
      waveform [B, T] — 24kHz 파형

    핵심 설계:
    1. Content + Timbre + Pitch를 결합 (timbre는 시간축 broadcast)
    2. Causal Upsample Blocks로 프레임 → 샘플 단위 확장
    3. ConvNeXt refinement로 세부 특징 정제
    4. iSTFT 헤드로 파형 합성 (Vocos 방식)

    파라미터: ~1.5M (INT8 ONNX: ~1.5MB)
    """

    def __init__(
        self,
        content_dim: int = 64,
        timbre_dim: int = 128,
        pitch_dim: int = 1,
        hidden_channels: int = 256,
        # 인코더 strides=(4,4,5,3) → 240배 다운샘플
        # 디코더는 iSTFT hop=240으로 역변환 → CNN 업샘플은 필요 없음
        # 다만 프레임 수는 인코더와 동일 (100fps → iSTFT로 24kHz)
        n_fft: int = 480,
        hop_length: int = 240,
        n_blocks: int = 6,
        use_pitch: bool = True,
    ):
        super().__init__()
        self.use_pitch = use_pitch
        self.hop_length = hop_length

        # === 입력 융합 ===
        # content + broadcast(timbre) + pitch → hidden
        in_dim = content_dim + timbre_dim + (pitch_dim if use_pitch else 0)
        self.input_proj = nn.Conv1d(in_dim, hidden_channels, 1)

        # === ConvNeXt 정제 블록 (프레임 레벨) ===
        # Vocos처럼 프레임 단위로 충분히 정제 후 iSTFT로 한번에 샘플 확장
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(hidden_channels, kernel_size=7, expand_ratio=3)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.BatchNorm1d(hidden_channels)

        # === iSTFT 헤드 ===
        self.istft_head = ISTFTHead(
            hidden_channels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(
        self,
        content: torch.Tensor,
        timbre: torch.Tensor,
        pitch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            content: [B, content_dim, T']
            timbre:  [B, timbre_dim]
            pitch:   [B, 1, T'] — 옵션 (use_pitch=True일 때)

        Returns:
            waveform: [B, T] — 24kHz 파형
        """
        B, _, T_frames = content.shape

        # timbre를 시간축으로 broadcast
        timbre_t = timbre.unsqueeze(-1).expand(-1, -1, T_frames)  # [B, timbre_dim, T']

        # 입력 융합
        if self.use_pitch:
            if pitch is None:
                pitch = torch.zeros(B, 1, T_frames, device=content.device)
            x = torch.cat([content, timbre_t, pitch], dim=1)
        else:
            x = torch.cat([content, timbre_t], dim=1)

        x = self.input_proj(x)  # [B, hidden, T']

        # 정제
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # iSTFT로 파형 합성
        waveform = self.istft_head(x)  # [B, T]

        return waveform
