"""
Phase 3: 경량화 & 스트리밍 증류 (LLVC 방식).

Teacher(Phase 2의 큰 모델)의 출력을 student(더 작은 모델)가 모방.
- Student는 parameter가 더 적고, 연산이 더 가벼움
- Teacher의 변환된 파형 + content feature 모두 매칭
- 최종 ONNX 변환 대상
"""

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..models import SoramiPipeline, CombinedDiscriminator
from ..losses import (
    MultiResMelLoss,
    discriminator_loss,
    generator_loss,
    feature_matching_loss,
)
from ..data import AudioDataset


class Phase3Trainer(BaseTrainer):
    def build_models(self):
        mcfg = self.config["model"]
        # Student: 경량 설정
        student = SoramiPipeline(
            content_dim=mcfg["content_dim"],
            timbre_dim=mcfg["timbre_dim"],
            hidden_channels=mcfg["hidden_channels"],
            n_targets=mcfg["n_targets"],
            use_pitch=mcfg["use_pitch"],
            sample_rate=mcfg["sample_rate"],
        )
        discriminator = CombinedDiscriminator()
        return student, discriminator

    def build_dataset(self):
        dcfg = self.config["data"]
        return AudioDataset(
            root=dcfg["root"],
            sample_rate=self.config["model"]["sample_rate"],
            segment_seconds=dcfg["segment_seconds"],
            augment=dcfg["augment"],
        )

    def _load_teacher(self):
        """Teacher 모델을 freeze 상태로 로드."""
        tcfg = self.config["teacher"]
        teacher_ckpt = torch.load(tcfg["ckpt"], map_location=self.device)
        teacher_config = teacher_ckpt["config"]["model"]
        teacher = SoramiPipeline(
            content_dim=teacher_config["content_dim"],
            timbre_dim=teacher_config["timbre_dim"],
            hidden_channels=teacher_config["hidden_channels"],
            n_targets=teacher_config["n_targets"],
            use_pitch=teacher_config["use_pitch"],
            sample_rate=teacher_config["sample_rate"],
        )
        teacher.load_state_dict(teacher_ckpt["generator"])
        teacher.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def setup(self):
        super().setup()
        sr = self.config["model"]["sample_rate"]
        self.mel_loss = MultiResMelLoss(sample_rate=sr).to(self.device)
        self.teacher = self._load_teacher()

    def train_step(self, batch: dict) -> dict:
        wav = batch["waveform"]                   # [B, 1, T]
        speaker_id = batch["speaker_id"]
        target_id = speaker_id.clamp(0, self.config["model"]["n_targets"] - 1)

        # Teacher 출력 (오차 역전파 안 함)
        with torch.no_grad():
            teacher_out = self.teacher(wav, target_id=target_id)
            teacher_wav = teacher_out["waveform"]

        # Student 출력
        self.generator.train()
        student_out = self.generator(wav, target_id=target_id)
        student_wav = student_out["waveform"]

        # 길이 정렬
        T = min(teacher_wav.size(-1), student_wav.size(-1))
        teacher_wav = teacher_wav[..., :T]
        student_wav = student_wav[..., :T]

        teacher_wav_3d = teacher_wav.unsqueeze(1) if teacher_wav.dim() == 2 else teacher_wav
        student_wav_3d = student_wav.unsqueeze(1) if student_wav.dim() == 2 else student_wav

        # ========== Discriminator (teacher를 real로 간주) ==========
        self.opt_d.zero_grad()
        real_logits, _ = self.discriminator(teacher_wav_3d)
        fake_logits, _ = self.discriminator(student_wav_3d.detach())
        d_loss, _, _ = discriminator_loss(real_logits, fake_logits)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.config["train"].get("grad_clip", 1.0),
        )
        self.opt_d.step()

        # ========== Student 학습 ==========
        self.opt_g.zero_grad()

        # Teacher 파형 모방 (L1 on waveform — LLVC의 핵심)
        wav_l1 = F.l1_loss(student_wav, teacher_wav)

        # Mel loss (teacher vs student)
        mel_l = self.mel_loss(student_wav, teacher_wav)

        # Adversarial + FM
        real_logits, real_fmaps = self.discriminator(teacher_wav_3d)
        fake_logits, fake_fmaps = self.discriminator(student_wav_3d)
        adv_l, _ = generator_loss(fake_logits)
        fm_l = feature_matching_loss(real_fmaps, fake_fmaps)

        # Content feature 매칭 (teacher의 content feature를 student가 따라가도록)
        with torch.no_grad():
            t_content, _ = self.teacher.encoder(wav)
        s_content, _ = self.generator.encoder(wav)
        # 차원이 다를 수 있음 (student가 더 작음) → 마지막 채널 수만 비교
        min_c = min(t_content.size(1), s_content.size(1))
        min_T2 = min(t_content.size(-1), s_content.size(-1))
        content_l = F.l1_loss(
            s_content[:, :min_c, :min_T2],
            t_content[:, :min_c, :min_T2],
        )

        lcfg = self.config["loss"]
        total_g = (
            lcfg["lambda_mel"] * mel_l
            + lcfg["lambda_adv"] * adv_l
            + lcfg["lambda_fm"] * fm_l
            + lcfg["lambda_content"] * content_l
            + lcfg.get("lambda_wav_l1", 10.0) * wav_l1
        )

        total_g.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.config["train"].get("grad_clip", 1.0),
        )
        self.opt_g.step()

        return {
            "wav_l1": wav_l1,
            "mel": mel_l,
            "adv": adv_l,
            "fm": fm_l,
            "content": content_l,
            "total_g": total_g,
            "d_loss": d_loss,
        }
