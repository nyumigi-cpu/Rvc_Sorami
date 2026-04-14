"""
Content consistency & timbre contrastive losses.

- ContentConsistencyLoss: 변환 전/후 콘텐츠가 보존되도록 (cycle-like)
- TimbreContrastiveLoss: 같은 화자 → 비슷한 임베딩, 다른 화자 → 다른 임베딩

DeCodec 영감: 콘텐츠와 음색을 수학적으로 분리하는 핵심 손실.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentConsistencyLoss(nn.Module):
    """
    변환된 파형을 다시 인코딩했을 때 콘텐츠가 보존되도록.

    L_content = || Encoder(convert(x)).content - Encoder(x).content ||₁

    이렇게 하면 timbre adapter가 콘텐츠를 손상시키지 않게 강제된다.
    (비언어 소리의 구조 보존에 특히 중요)
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, content_original: torch.Tensor,
                content_reconstructed: torch.Tensor) -> torch.Tensor:
        # 시간축 길이 정렬
        t = min(content_original.size(-1), content_reconstructed.size(-1))
        c_orig = content_original[..., :t]
        c_rec = content_reconstructed[..., :t]

        if self.loss_type == "l1":
            return F.l1_loss(c_rec, c_orig.detach())
        elif self.loss_type == "l2":
            return F.mse_loss(c_rec, c_orig.detach())
        elif self.loss_type == "cosine":
            # 채널 방향 코사인 유사도 → 1에 가깝게
            cos = F.cosine_similarity(c_rec, c_orig.detach(), dim=1)
            return 1.0 - cos.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class TimbreContrastiveLoss(nn.Module):
    """
    InfoNCE 스타일 대조 손실.

    같은 화자의 서로 다른 발화에서 추출한 timbre는 유사 → positive
    다른 화자의 timbre는 멀리 → negative

    DeCodec 방식에서 콘텐츠/음색 분리의 핵심: 콘텐츠 정보가 timbre에 새지 않게 함.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, timbres: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timbres:     [B, D] — 배치 내 timbre embedding들
            speaker_ids: [B]    — 각 샘플의 화자 ID
        Returns:
            loss: scalar
        """
        B = timbres.size(0)
        if B < 2:
            return torch.zeros((), device=timbres.device, requires_grad=True)

        # L2 normalize
        t = F.normalize(timbres, dim=-1)
        # Similarity matrix [B, B]
        sim = torch.matmul(t, t.t()) / self.temperature

        # Self-similarity 제외 (대각선 마스크)
        mask_self = torch.eye(B, dtype=torch.bool, device=timbres.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Positive pair: 같은 화자
        sid = speaker_ids.view(-1, 1)
        pos_mask = (sid == sid.t()) & (~mask_self)

        # 각 anchor에 대해 positive가 하나라도 있는 경우만 계산
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.zeros((), device=timbres.device, requires_grad=True)

        # log-softmax over all others, sum over positives
        log_prob = F.log_softmax(sim, dim=1)
        # 평균 log-prob over positives (각 anchor별)
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        loss = -pos_log_prob[has_pos].mean()
        return loss
