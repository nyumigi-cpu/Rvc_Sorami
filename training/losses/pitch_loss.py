"""
Pitch estimation loss.

- Cross-entropy on pitch bin distribution
- BCE on voicing
- L1 on predicted Hz (유성 구간만)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.pitch_estimator import hz_to_bins, N_PITCH_BINS


class PitchLoss(nn.Module):
    """피치 추정기 학습용 손실.

    타겟:
      - target_pitch_hz: [B, 1, T'] — 외부 도구(예: pyworld)로 추출한 F0
      - target_voicing:  [B, 1, T'] — 유성(1) / 무성(0) 마스크
    """

    def __init__(self, alpha_voicing: float = 1.0, alpha_hz: float = 0.1):
        super().__init__()
        self.alpha_voicing = alpha_voicing
        self.alpha_hz = alpha_hz

    def forward(
        self,
        pitch_logits: torch.Tensor,
        voicing_logits: torch.Tensor,
        target_pitch_hz: torch.Tensor,
        target_voicing: torch.Tensor,
    ) -> dict:
        """
        Args:
            pitch_logits:   [B, N_BINS, T']
            voicing_logits: [B, 1, T']
            target_pitch_hz: [B, 1, T']
            target_voicing:  [B, 1, T']

        Returns:
            dict: {"pitch": ..., "voicing": ..., "hz": ..., "total": ...}
        """
        # 시간축 정렬
        t = min(pitch_logits.size(-1), target_pitch_hz.size(-1))
        pitch_logits = pitch_logits[..., :t]
        voicing_logits = voicing_logits[..., :t]
        target_pitch_hz = target_pitch_hz[..., :t]
        target_voicing = target_voicing[..., :t]

        # Bin 타겟 (유성 구간만)
        target_bins = hz_to_bins(target_pitch_hz.squeeze(1))  # [B, T']
        target_bins = target_bins.clamp(0, N_PITCH_BINS - 1).long()

        voiced_mask = target_voicing.squeeze(1) > 0.5

        # Pitch cross-entropy (유성 구간)
        if voiced_mask.any():
            pitch_ce = F.cross_entropy(
                pitch_logits.transpose(1, 2).reshape(-1, N_PITCH_BINS)[voiced_mask.view(-1)],
                target_bins.view(-1)[voiced_mask.view(-1)],
            )
        else:
            pitch_ce = torch.zeros((), device=pitch_logits.device, requires_grad=True)

        # Voicing BCE
        voicing_bce = F.binary_cross_entropy_with_logits(voicing_logits, target_voicing)

        # Hz L1 (보조, 유성 구간)
        # soft argmax로 예측 Hz 계산
        from ..models.pitch_estimator import decode_pitch
        pred_hz = decode_pitch(pitch_logits, voicing_logits)
        if voiced_mask.any():
            hz_l1 = F.l1_loss(
                pred_hz[voiced_mask.unsqueeze(1)],
                target_pitch_hz[voiced_mask.unsqueeze(1)],
            )
        else:
            hz_l1 = torch.zeros((), device=pitch_logits.device, requires_grad=True)

        total = pitch_ce + self.alpha_voicing * voicing_bce + self.alpha_hz * hz_l1
        return {
            "pitch": pitch_ce,
            "voicing": voicing_bce,
            "hz": hz_l1,
            "total": total,
        }
