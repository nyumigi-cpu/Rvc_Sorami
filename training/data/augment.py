"""
Data augmentation for robustness.

- Gain jitter, noise injection, time masking
- 목표: 실제 게임/통화 환경(노이즈, 음량 변동)에서도 작동하게
"""

import torch
import torch.nn.functional as F
import random


class RandomAugment:
    """학습 데이터 증강. inference-time에는 사용하지 않음.

    적용:
    - gain_jitter: 음량 랜덤 변동 (±6dB)
    - noise: 백색/마이크 노이즈 추가
    - time_mask: 짧은 구간 무음 (robust한 콘텐츠 표현 유도)
    - pitch_jitter: 학습 증강용 ±2 semitone (소스에만, 타겟은 보존)
    """

    def __init__(
        self,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        noise_snr_range: tuple[float, float] = (20.0, 40.0),
        noise_prob: float = 0.3,
        time_mask_prob: float = 0.1,
        time_mask_ratio: float = 0.02,
        sample_rate: int = 24000,
    ):
        self.gain_db_range = gain_db_range
        self.noise_snr_range = noise_snr_range
        self.noise_prob = noise_prob
        self.time_mask_prob = time_mask_prob
        self.time_mask_ratio = time_mask_ratio
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, T]
        Returns:
            augmented: [1, T]
        """
        x = waveform.clone()

        # Gain jitter
        db = random.uniform(*self.gain_db_range)
        x = x * (10 ** (db / 20.0))

        # Noise injection
        if random.random() < self.noise_prob:
            snr_db = random.uniform(*self.noise_snr_range)
            signal_pow = x.pow(2).mean().clamp(min=1e-10)
            noise_pow = signal_pow / (10 ** (snr_db / 10.0))
            noise = torch.randn_like(x) * noise_pow.sqrt()
            x = x + noise

        # Time masking (짧은 무음)
        if random.random() < self.time_mask_prob:
            T = x.size(-1)
            mask_len = int(T * self.time_mask_ratio)
            if mask_len > 0:
                start = random.randint(0, T - mask_len)
                x[..., start : start + mask_len] = 0.0

        # 클리핑 방지
        peak = x.abs().max()
        if peak > 0.99:
            x = x * (0.99 / peak)

        return x
