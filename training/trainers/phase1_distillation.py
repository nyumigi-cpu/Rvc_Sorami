"""
Phase 1: Teacher 증류 학습.

외부 Teacher 모델(Seed-VC / Takin-VC)이 미리 생성한 (source_male, target_female) 병렬 쌍을 사용.
Sorami 파이프라인이 소스를 타겟과 비슷하게 변환하도록 재구성 손실로 학습.
GAN 없이 안정적으로 수렴 — 타겟 음색 임베딩 + 피치 매핑 초기화 목적.
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..models import SoramiPipeline
from ..losses import MultiResMelLoss, ContentConsistencyLoss
from ..data import ParallelAudioDataset


class Phase1Trainer(BaseTrainer):
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
        # Phase 1은 GAN 없음
        return generator, None

    def build_dataset(self):
        dcfg = self.config["data"]
        return ParallelAudioDataset(
            manifest=dcfg["manifest"],
            sample_rate=self.config["model"]["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            augment=dcfg["augment"],
        )

    def setup(self):
        super().setup()
        # 손실 함수들
        self.mel_loss = MultiResMelLoss(
            sample_rate=self.config["model"]["sample_rate"]
        ).to(self.device)
        self.content_loss = ContentConsistencyLoss(loss_type="l1").to(self.device)

    def train_step(self, batch: dict) -> dict:
        src = batch["source"]           # [B, 1, T]
        tgt = batch["target"]           # [B, 1, T]
        target_id = batch["target_id"]  # [B]

        self.generator.train()

        # Forward
        out = self.generator(src, target_id=target_id)
        pred_wav = out["waveform"]  # [B, T']

        # Target을 2D로
        tgt_wav = tgt.squeeze(1)  # [B, T]

        # Mel reconstruction loss
        mel_l = self.mel_loss(pred_wav, tgt_wav)

        # Content consistency: pred를 다시 인코딩 → content가 원본 src content와 유사해야
        # (timbre adapter가 content를 망치지 않게)
        pred_wav_in = pred_wav.unsqueeze(1) if pred_wav.dim() == 2 else pred_wav
        # 길이 맞춤
        min_T = min(pred_wav_in.size(-1), src.size(-1))
        pred_enc = self.generator.encoder(pred_wav_in[..., :min_T])
        src_enc = self.generator.encoder(src[..., :min_T])
        content_l = self.content_loss(src_enc[0], pred_enc[0])

        # 총 손실
        lcfg = self.config["loss"]
        total = lcfg["lambda_mel"] * mel_l + lcfg["lambda_content"] * content_l

        self.opt_g.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.config["train"].get("grad_clip", 1.0),
        )
        self.opt_g.step()

        return {
            "mel": mel_l,
            "content": content_l,
            "total": total,
        }
