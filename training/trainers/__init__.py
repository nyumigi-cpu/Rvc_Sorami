"""Training loops for Phase 1 (distillation), Phase 2 (GAN), Phase 3 (compression)."""

from .base_trainer import BaseTrainer, load_config
from .phase1_distillation import Phase1Trainer
from .phase2_gan import Phase2Trainer
from .phase3_distill import Phase3Trainer

__all__ = [
    "BaseTrainer",
    "load_config",
    "Phase1Trainer",
    "Phase2Trainer",
    "Phase3Trainer",
]
