"""
Timbre Adapter — 음색 교체 네트워크

설계 근거:
- VChangeCodec 방식: 코덱 인코더에 음색 적응 네트워크 내장
- AdaIN (Adaptive Instance Normalization): 화자 임베딩 → scale/shift로 음색 주입
  → Cross-Attention O(n²)보다 훨씬 빠른 O(n) 연산
- DeCodec 방식: 직교 투영(SOP)으로 콘텐츠와 음색을 수학적으로 분리
- StyleStream 방식: 화자 임베딩이 모든 레이어에 걸쳐 조건화

파이프라인:
  Source content [B, C, T] + Target timbre [B, D]
    → FiLM/AdaIN 변조로 타겟 화자의 음색 주입
    → 콘텐츠 구조는 보존, 음색 특성만 교체
    → Adapted content [B, C, T]

핵심: 원본 화자의 음색 흔적을 지우고, 타겟 화자의 음색을 주입.
     콘텐츠(발음/웃음/한숨 구조)는 절대 건드리지 않음.

목표: ~0.3M 파라미터, CPU에서 2-3ms 추론
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec_encoder import CausalConv1d


class OrthogonalProjection(nn.Module):
    """직교 투영 (DeCodec 방식) — 콘텐츠에서 음색 성분 제거.

    원리: 음색 벡터 t가 주어지면, 콘텐츠 c에서 t 방향 성분을 제거
          c' = c - (c · t_hat) * t_hat
    이후 타겟 음색 t'를 더함.

    실제로는 학습 가능한 투영 행렬을 써서 더 유연하게.
    """

    def __init__(self, content_dim: int, timbre_dim: int):
        super().__init__()
        # 음색 → 콘텐츠 공간 투영
        self.timbre_to_content = nn.Linear(timbre_dim, content_dim, bias=False)
        # 제거 강도 (학습 가능)
        self.removal_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, content: torch.Tensor, src_timbre: torch.Tensor,
                tgt_timbre: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content:    [B, C, T]  — 소스 콘텐츠 (음색 포함)
            src_timbre: [B, D]     — 소스 화자 음색
            tgt_timbre: [B, D]     — 타겟 화자 음색

        Returns:
            content: [B, C, T] — 음색 교체된 콘텐츠
        """
        # 음색을 콘텐츠 공간으로 투영
        src_vec = self.timbre_to_content(src_timbre).unsqueeze(-1)  # [B, C, 1]
        tgt_vec = self.timbre_to_content(tgt_timbre).unsqueeze(-1)  # [B, C, 1]

        # src 음색 방향 성분 제거 → tgt 음색 주입
        # 단순 덧셈/뺄셈 대신 학습 가능한 스케일로 부드럽게
        content = content - self.removal_scale * src_vec + self.removal_scale * tgt_vec
        return content


class FiLMLayer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) — 화자 임베딩으로 scale/shift.

    AdaIN과 유사하나 더 단순. 각 채널별로 독립된 γ, β를 화자 벡터에서 생성.
    """

    def __init__(self, channels: int, timbre_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(timbre_dim, channels)
        self.to_shift = nn.Linear(timbre_dim, channels)
        # 초기값: scale=1, shift=0이 되도록
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, timbre: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      [B, C, T]
            timbre: [B, D]
        Returns:
            [B, C, T]
        """
        scale = self.to_scale(timbre).unsqueeze(-1)  # [B, C, 1]
        shift = self.to_shift(timbre).unsqueeze(-1)  # [B, C, 1]
        # 초기엔 변조 없음 (1+scale로 start from identity)
        return x * (1.0 + scale) + shift


class AdaINBlock(nn.Module):
    """AdaIN 블록 — Instance Norm 후 FiLM 변조.

    구조:
        DWConv → InstanceNorm → FiLM(timbre) → GELU → PWConv → residual
    """

    def __init__(self, channels: int, timbre_dim: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = CausalConv1d(channels, channels, kernel_size, groups=channels)
        # InstanceNorm으로 화자 정보 정규화 (제거), 이후 FiLM으로 타겟 음색 주입
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.film = FiLMLayer(channels, timbre_dim)
        self.pwconv = nn.Conv1d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, timbre: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)            # 채널별 정규화 → 원본 음색 통계 제거
        x = self.film(x, timbre)    # 타겟 음색 통계 주입
        x = self.act(x)
        x = self.pwconv(x)
        return x + residual


class TimbreAdapter(nn.Module):
    """
    음색 교체 네트워크.

    입력:
      - content     [B, content_dim, T'] — 소스 콘텐츠 특징
      - src_timbre  [B, timbre_dim]      — 소스 화자 임베딩 (인코더에서 나옴)
      - tgt_timbre  [B, timbre_dim]      — 타겟 화자 임베딩 (학습된 또는 저장된)

    출력:
      - adapted_content [B, content_dim, T'] — 타겟 음색이 주입된 콘텐츠

    핵심 설계:
    1. 직교 투영으로 소스 음색 흔적 제거
    2. AdaIN 블록 여러 개로 타겟 음색 점진적 주입
    3. 콘텐츠 구조(시간축 패턴)는 변경 최소화 → residual로 보존

    파라미터: ~0.3M (INT8 ONNX: ~0.3MB)
    """

    def __init__(
        self,
        content_dim: int = 64,
        timbre_dim: int = 128,
        hidden_channels: int = 128,
        n_blocks: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()

        # === Step 1: 직교 투영으로 소스 음색 제거 + 타겟 음색 초기 주입 ===
        self.projection = OrthogonalProjection(content_dim, timbre_dim)

        # === Step 2: 차원 확장 ===
        self.input_proj = nn.Conv1d(content_dim, hidden_channels, 1)

        # === Step 3: AdaIN 블록들 (타겟 음색 점진적 주입) ===
        self.blocks = nn.ModuleList([
            AdaINBlock(hidden_channels, timbre_dim, kernel_size=kernel_size)
            for _ in range(n_blocks)
        ])

        # === Step 4: 차원 축소 (원래 content_dim으로 복원) ===
        self.output_proj = nn.Conv1d(hidden_channels, content_dim, 1)

    def forward(
        self,
        content: torch.Tensor,
        src_timbre: torch.Tensor,
        tgt_timbre: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            content:    [B, content_dim, T']
            src_timbre: [B, timbre_dim]
            tgt_timbre: [B, timbre_dim]

        Returns:
            adapted: [B, content_dim, T']
        """
        # Step 1: 음색 교체 (직교 투영)
        x = self.projection(content, src_timbre, tgt_timbre)

        # Step 2-3: AdaIN 블록들로 타겟 음색 점진적 주입
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, tgt_timbre)

        # Step 4: 복원
        x = self.output_proj(x)

        # 원본 콘텐츠 구조 보존을 위한 residual
        # (음색만 교체하고 콘텐츠 시간 구조는 유지)
        return content + x


class TargetTimbreBank(nn.Module):
    """타겟 화자 음색 뱅크 — 학습 가능한 여성 화자 임베딩.

    1:1 전용 모델이므로 타겟 화자는 1명 (또는 여러 스타일).
    추론 시에는 이 임베딩을 직접 사용.
    """

    def __init__(self, timbre_dim: int = 128, n_speakers: int = 1):
        super().__init__()
        self.n_speakers = n_speakers
        # 학습 가능한 타겟 화자 임베딩
        # 초기화: 작은 정규분포 (너무 크면 초기 학습 불안정)
        self.embeddings = nn.Parameter(torch.randn(n_speakers, timbre_dim) * 0.1)

    def forward(self, speaker_id: torch.Tensor | int = 0) -> torch.Tensor:
        """
        Args:
            speaker_id: [B] 또는 int. 여러 스타일 학습 시 사용.

        Returns:
            timbre: [B, timbre_dim]
        """
        if isinstance(speaker_id, int):
            return self.embeddings[speaker_id].unsqueeze(0)  # [1, D]
        return self.embeddings[speaker_id]  # [B, D]
