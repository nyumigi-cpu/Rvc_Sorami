"""
Adversarial losses (HiFi-GAN 스타일 least-squares GAN).

- discriminator_loss: D 학습 시 손실
- generator_loss: G 학습 시 adversarial 부분
- feature_matching_loss: D의 중간 feature map을 맞추는 지각적 손실
"""

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits: list, fake_logits: list) -> tuple[torch.Tensor, list, list]:
    """
    LSGAN 판별기 손실:
        L_D = E[(D(real) - 1)²] + E[D(fake)²]

    Args:
        real_logits: list of [B, *] — 각 sub-discriminator 출력
        fake_logits: list of [B, *]

    Returns:
        loss:       scalar — 총 손실
        r_losses:   list   — 각 D별 real loss (모니터링)
        f_losses:   list   — 각 D별 fake loss (모니터링)
    """
    loss = 0.0
    r_losses, f_losses = [], []
    for dr, df in zip(real_logits, fake_logits):
        r_loss = torch.mean((dr - 1.0) ** 2)
        f_loss = torch.mean(df ** 2)
        loss = loss + r_loss + f_loss
        r_losses.append(r_loss.item())
        f_losses.append(f_loss.item())
    return loss, r_losses, f_losses


def generator_loss(fake_logits: list) -> tuple[torch.Tensor, list]:
    """
    LSGAN 생성기 손실:
        L_G = E[(D(fake) - 1)²]
    """
    loss = 0.0
    g_losses = []
    for df in fake_logits:
        l = torch.mean((df - 1.0) ** 2)
        loss = loss + l
        g_losses.append(l.item())
    return loss, g_losses


def feature_matching_loss(real_fmaps: list, fake_fmaps: list) -> torch.Tensor:
    """
    Feature matching loss (HiFi-GAN):
        각 D의 중간 layer feature map들을 L1으로 매칭.
        mel loss와 함께 파형 디테일 복원에 중요.

    Args:
        real_fmaps: list of list — [discriminator][layer] 형태
        fake_fmaps: same shape

    Returns:
        loss: scalar
    """
    loss = 0.0
    count = 0
    for d_real, d_fake in zip(real_fmaps, fake_fmaps):
        for r, f in zip(d_real, d_fake):
            loss = loss + F.l1_loss(f, r.detach())
            count += 1
    return loss / max(count, 1)
