"""
Sorami Pipeline — 전체 모듈 결합

입력: 소스 남성 음성 [B, 1, T] (24kHz)
출력: 변환된 여성 음성 [B, 1, T]

파이프라인:
  source_wav
    ├→ [CodecEncoder] → content, src_timbre
    ├→ [PitchEstimator] → src_pitch → shifted_pitch (남→여)
    │
    └→ [TimbreAdapter(content, src_timbre, tgt_timbre)] → adapted_content
        └→ [CodecDecoder(adapted_content, tgt_timbre, shifted_pitch)] → converted_wav

추론 모드:
  - target_id로 타겟 화자 선택
  - 모든 모듈 인과적이라 스트리밍 가능
"""

import torch
import torch.nn as nn

from .codec_encoder import CodecEncoder
from .codec_decoder import CodecDecoder
from .timbre_adapter import TimbreAdapter, TargetTimbreBank
from .pitch_estimator import PitchEstimator


class SoramiPipeline(nn.Module):
    """
    전체 Sorami 파이프라인.

    두 가지 모드:
    - forward(source_wav, target_id): 학습/추론용 end-to-end
    - forward_with_target_timbre(source_wav, target_timbre): 타겟 음색을 직접 지정
    """

    def __init__(
        self,
        content_dim: int = 64,
        timbre_dim: int = 128,
        hidden_channels: int = 256,
        n_targets: int = 1,
        use_pitch: bool = True,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_pitch = use_pitch
        self.n_targets = n_targets

        # 핵심 모듈들
        self.encoder = CodecEncoder(
            hidden_channels=hidden_channels,
            content_dim=content_dim,
            timbre_dim=timbre_dim,
        )
        self.timbre_adapter = TimbreAdapter(
            content_dim=content_dim,
            timbre_dim=timbre_dim,
        )
        self.pitch_estimator = PitchEstimator(n_targets=n_targets)
        self.decoder = CodecDecoder(
            content_dim=content_dim,
            timbre_dim=timbre_dim,
            hidden_channels=hidden_channels,
            use_pitch=use_pitch,
        )

        # 타겟 화자 음색 뱅크 (학습 가능)
        self.target_bank = TargetTimbreBank(timbre_dim=timbre_dim, n_speakers=n_targets)

    def encode(self, waveform: torch.Tensor) -> dict:
        """인코딩만 수행. (전체 파이프라인의 앞부분)"""
        content, src_timbre = self.encoder(waveform)
        out = {"content": content, "src_timbre": src_timbre}
        if self.use_pitch:
            shifted_pitch, voicing = self.pitch_estimator(waveform, target_id=0)
            # 시간축 길이를 content와 맞춤 (둘 다 100fps지만 경계 차이 방지)
            shifted_pitch = self._align_time(shifted_pitch, content.size(-1))
            voicing = self._align_time(voicing, content.size(-1))
            out["shifted_pitch"] = shifted_pitch
            out["voicing"] = voicing
        return out

    def decode(
        self,
        content: torch.Tensor,
        src_timbre: torch.Tensor,
        tgt_timbre: torch.Tensor,
        pitch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """콘텐츠 + 음색 교체 + 디코딩. (파이프라인 뒷부분)"""
        adapted = self.timbre_adapter(content, src_timbre, tgt_timbre)
        waveform = self.decoder(adapted, tgt_timbre, pitch=pitch)
        return waveform

    def forward(
        self,
        source_wav: torch.Tensor,
        target_id: int | torch.Tensor = 0,
    ) -> dict:
        """
        End-to-end forward.

        Args:
            source_wav: [B, 1, T] — 소스 남성 음성 (24kHz)
            target_id:  int or [B] — 타겟 화자 ID

        Returns:
            dict {
                'waveform':      [B, T'] — 변환된 파형
                'content':       [B, C, T'],
                'src_timbre':    [B, D],
                'tgt_timbre':    [B, D],
                'shifted_pitch': [B, 1, T'] or None,
                'voicing':       [B, 1, T'] or None,
            }
        """
        B = source_wav.size(0)

        enc = self.encode(source_wav)

        # 타겟 음색 조회
        if isinstance(target_id, int):
            tgt_timbre = self.target_bank.embeddings[target_id].unsqueeze(0).expand(B, -1)
        else:
            tgt_timbre = self.target_bank(target_id)

        waveform = self.decode(
            enc["content"],
            enc["src_timbre"],
            tgt_timbre,
            pitch=enc.get("shifted_pitch"),
        )

        return {
            "waveform": waveform,
            "content": enc["content"],
            "src_timbre": enc["src_timbre"],
            "tgt_timbre": tgt_timbre,
            "shifted_pitch": enc.get("shifted_pitch"),
            "voicing": enc.get("voicing"),
        }

    def forward_with_target_timbre(
        self,
        source_wav: torch.Tensor,
        target_timbre: torch.Tensor,
    ) -> torch.Tensor:
        """타겟 음색 벡터를 직접 지정 (few-shot 적응용)."""
        enc = self.encode(source_wav)
        waveform = self.decode(
            enc["content"],
            enc["src_timbre"],
            target_timbre,
            pitch=enc.get("shifted_pitch"),
        )
        return waveform

    @staticmethod
    def _align_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """시간축 길이를 target_len에 맞춰 자르거나 우측 패딩."""
        cur = x.size(-1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[..., :target_len]
        # 짧으면 마지막 값 반복 패딩
        pad = target_len - cur
        last = x[..., -1:].expand(*x.shape[:-1], pad)
        return torch.cat([x, last], dim=-1)

    # === 추론 편의 함수 ===

    @torch.no_grad()
    def convert(
        self,
        source_wav: torch.Tensor,
        target_id: int = 0,
    ) -> torch.Tensor:
        """순수 변환 (eval 모드).

        Returns:
            waveform: [B, T'] — 24kHz 변환 파형
        """
        self.eval()
        out = self.forward(source_wav, target_id=target_id)
        return out["waveform"]
