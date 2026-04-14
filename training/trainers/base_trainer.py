"""
Base trainer — 공통 로직 (체크포인트, 로깅, 옵티마이저 설정).

각 Phase 트레이너는 이걸 상속해 train_step만 구현한다.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def load_config(path: str | Path) -> dict:
    """YAML 설정 로딩. base.yaml을 상속하는 경우 병합."""
    import yaml

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # base.yaml 자동 상속
    base_path = path.parent / "base.yaml"
    if base_path.exists() and path.name != "base.yaml":
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """중첩 dict 병합 (override가 우선)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseTrainer:
    """
    공통 트레이너 베이스.

    서브클래스는 반드시 구현해야 함:
      - build_models() → (generator, discriminator or None)
      - build_dataset() → dataset
      - train_step(batch) → dict of losses
    """

    def __init__(self, config: dict, device: str | None = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.get("seed", 42))

        self.output_dir = Path(config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_loss = float("inf")

    # === 서브클래스가 구현해야 하는 메서드 ===

    def build_models(self) -> tuple[nn.Module, nn.Module | None]:
        raise NotImplementedError

    def build_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def train_step(self, batch: dict) -> dict:
        raise NotImplementedError

    # === 공통 setup ===

    def build_optimizers(self):
        tcfg = self.config["train"]
        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=tcfg["lr_g"],
            betas=tcfg.get("betas", [0.8, 0.99]),
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        if self.discriminator is not None:
            self.opt_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=tcfg.get("lr_d", tcfg["lr_g"]),
                betas=tcfg.get("betas", [0.8, 0.99]),
                weight_decay=tcfg.get("weight_decay", 0.01),
            )
        else:
            self.opt_d = None

    def setup(self):
        """전체 초기화."""
        self.generator, self.discriminator = self.build_models()
        self.generator.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)

        self.dataset = self.build_dataset()
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"].get("num_workers", 4),
            pin_memory=self.config["data"].get("pin_memory", True),
            drop_last=True,
        )

        self.build_optimizers()

        # Pretrained 체크포인트 로딩
        pretrained = self.config.get("paths", {}).get("pretrained_ckpt")
        if pretrained and Path(pretrained).exists():
            self.load_checkpoint(pretrained, load_optimizers=False)
            print(f"[setup] Loaded pretrained from {pretrained}")

    # === 체크포인트 ===

    def save_checkpoint(self, name: str = "latest"):
        ckpt_path = self.output_dir / f"{name}.pt"
        state = {
            "step": self.step,
            "generator": self.generator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
            state["opt_d"] = self.opt_d.state_dict()
        torch.save(state, ckpt_path)
        print(f"[ckpt] Saved to {ckpt_path}")

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True):
        state = torch.load(path, map_location=self.device)
        # strict=False로 미세 차이 허용 (Phase 전환 시)
        missing, unexpected = self.generator.load_state_dict(
            state["generator"], strict=False,
        )
        if missing or unexpected:
            print(f"[ckpt] Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if load_optimizers and "opt_g" in state:
            self.opt_g.load_state_dict(state["opt_g"])
        if self.discriminator is not None and "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"], strict=False)
            if load_optimizers and "opt_d" in state:
                self.opt_d.load_state_dict(state["opt_d"])
        self.step = state.get("step", 0)
        self.best_loss = state.get("best_loss", float("inf"))

    # === 로깅 ===

    def log(self, losses: dict, t0: float):
        elapsed = time.time() - t0
        msg_parts = [f"step {self.step:>6d}"]
        msg_parts.append(f"t={elapsed:.2f}s")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            msg_parts.append(f"{k}={v:.4f}")
        print(" | ".join(msg_parts), flush=True)

    # === 메인 루프 ===

    def train(self):
        self.setup()
        tcfg = self.config["train"]
        max_steps = tcfg["max_steps"]
        log_every = tcfg.get("log_every", 100)
        ckpt_every = tcfg.get("ckpt_every", 5000)

        print(f"[train] Starting. max_steps={max_steps}")
        data_iter = iter(self.loader)

        while self.step < max_steps:
            t0 = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            batch = self._to_device(batch)
            losses = self.train_step(batch)
            self.step += 1

            if self.step % log_every == 0:
                self.log(losses, t0)

            if self.step % ckpt_every == 0:
                self.save_checkpoint("latest")
                # best 업데이트
                total = losses.get("total_g", losses.get("total", 0.0))
                if isinstance(total, torch.Tensor):
                    total = total.item()
                if total < self.best_loss:
                    self.best_loss = total
                    self.save_checkpoint("best")

        self.save_checkpoint("final")
        print("[train] Done.")

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
