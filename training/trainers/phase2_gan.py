"""
Phase 2: 코덱 + VC GAN 학습.

핵심 아이디어: 타겟 여성 오디오만 있으면 된다.
- 타겟 오디오를 자기 자신으로 reconstruct (self-recon) 학습
- 동시에 콘텐츠 분리를 위한 swap augmentation (같은 배치 내 shuffle)
- MPD + MSD 판별기로 자연스러운 파형 생성

학습 흐름:
  1. Generator forward: wav → (content, timbre) → (target_timbre로 swap) → reconstructed_wav
  2. Self-recon 경로: 같은 화자 원본을 target으로 하여 재구성
  3. Swap 경로: 다른 화자의 content를 target timbre로 변환 → discriminator만 피드백
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..models import SoramiPipeline, CombinedDiscriminator
from ..losses import (
    MultiResMelLoss,
    ContentConsistencyLoss,
    TimbreContrastiveLoss,
    discriminator_loss,
    generator_loss,
    feature_matching_loss,
)
from ..data import AudioDataset


class Phase2Trainer(BaseTrainer):
    def build_models(self):
        mcfg = self.config["model"]
        generator = SoramiPipeline(
            content_dim=mcfg["content_dim"],
            timbre_dim=mcfg["timbre_dim"],
            hidden_channels=mcfg["hidden_channels"],
            n_targets=mcfg["n_targets"],
            use_pitch=mcfg["use_pitch"],
            sample_rate=mcfg["sample_rate"],
        )
        discriminator = CombinedDiscriminator()
        return generator, discriminator

    def build_dataset(self):
        dcfg = self.config["data"]
        return AudioDataset(
            root=dcfg["root"],
            sample_rate=self.config["model"]["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            augment=dcfg["augment"],
            extract_pitch=dcfg.get("extract_pitch", False),
        )

    def setup(self):
        super().setup()
        sr = self.config["model"]["sample_rate"]
        self.mel_loss = MultiResMelLoss(sample_rate=sr).to(self.device)
        self.content_loss = ContentConsistencyLoss(loss_type="l1").to(self.device)
        self.timbre_contrastive = TimbreContrastiveLoss(temperature=0.1).to(self.device)

    def _align(self, pred_wav: torch.Tensor, target_wav: torch.Tensor):
        """pred와 target 길이 정렬."""
        if pred_wav.dim() == 2:
            pred_wav = pred_wav.unsqueeze(1)
        if target_wav.dim() == 2:
            target_wav = target_wav.unsqueeze(1)
        T = min(pred_wav.size(-1), target_wav.size(-1))
        return pred_wav[..., :T], target_wav[..., :T]

    def train_step(self, batch: dict) -> dict:
        wav = batch["waveform"]             # [B, 1, T]
        speaker_id = batch["speaker_id"]    # [B]

        # Self-reconstruction: 같은 화자 = target_id 0 (단일 타겟)
        # 다중 타겟이면 speaker_id를 타겟으로 사용
        target_id = speaker_id.clamp(0, self.config["model"]["n_targets"] - 1)

        # ========== 1) Generator forward ==========
        self.generator.train()
        out = self.generator(wav, target_id=target_id)
        pred_wav = out["waveform"]
        pred_wav, real_wav = self._align(pred_wav, wav)

        # ========== 2) Discriminator 학습 ==========
        if self.discriminator is not None:
            self.opt_d.zero_grad()
            real_logits, _ = self.discriminator(real_wav)
            fake_logits, _ = self.discriminator(pred_wav.detach())
            d_loss, _, _ = discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config["train"].get("grad_clip", 1.0),
            )
            self.opt_d.step()
        else:
            d_loss = torch.zeros((), device=self.device)

        # ========== 3) Generator 학습 ==========
        self.opt_g.zero_grad()

        # Mel loss (reconstruction)
        mel_l = self.mel_loss(pred_wav.squeeze(1), real_wav.squeeze(1))

        # Adversarial + Feature matching
        if self.discriminator is not None:
            real_logits, real_fmaps = self.discriminator(real_wav)
            fake_logits, fake_fmaps = self.discriminator(pred_wav)
            adv_l, _ = generator_loss(fake_logits)
            fm_l = feature_matching_loss(real_fmaps, fake_fmaps)
        else:
            adv_l = torch.zeros((), device=self.device)
            fm_l = torch.zeros((), device=self.device)

        # Content consistency: pred를 다시 인코딩해서 원본 content와 매칭
        with torch.no_grad():
            src_content, _ = self.generator.encoder(wav)
        pred_content, _ = self.generator.encoder(pred_wav)
        t = min(src_content.size(-1), pred_content.size(-1))
        content_l = F.l1_loss(pred_content[..., :t], src_content[..., :t])

        # Timbre contrastive: 같은 화자는 유사하게, 다른 화자는 멀게
        timbre_l = self.timbre_contrastive(out["src_timbre"], speaker_id)

        # 가중합
        lcfg = self.config["loss"]
        total_g = (
            lcfg["lambda_mel"] * mel_l
            + lcfg["lambda_adv"] * adv_l
            + lcfg["lambda_fm"] * fm_l
            + lcfg["lambda_content"] * content_l
            + lcfg["lambda_timbre"] * timbre_l
        )

        total_g.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.config["train"].get("grad_clip", 1.0),
        )
        self.opt_g.step()

        return {
            "mel": mel_l,
            "adv": adv_l,
            "fm": fm_l,
            "content": content_l,
            "timbre": timbre_l,
            "total_g": total_g,
            "d_loss": d_loss,
        }
