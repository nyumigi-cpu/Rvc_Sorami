"""
Sorami 통합 학습 스크립트.

사용법:
  python -m training.train --phase 1 --config training/configs/phase1_distillation.yaml
  python -m training.train --phase 2 --config training/configs/phase2_gan.yaml
  python -m training.train --phase 3 --config training/configs/phase3_distill.yaml
"""

import argparse
from pathlib import Path

from training.trainers import (
    load_config,
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
)


TRAINERS = {
    1: Phase1Trainer,
    2: Phase2Trainer,
    3: Phase3Trainer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로에서 재시작")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer_cls = TRAINERS[args.phase]
    trainer = trainer_cls(config=config, device=args.device)

    if args.resume:
        trainer.setup()
        trainer.load_checkpoint(args.resume, load_optimizers=True)
        print(f"[train] Resumed from step {trainer.step}")
        # 계속 학습
        tcfg = config["train"]
        import time
        from torch.utils.data import DataLoader
        data_iter = iter(trainer.loader)
        while trainer.step < tcfg["max_steps"]:
            t0 = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer.loader)
                batch = next(data_iter)
            batch = trainer._to_device(batch)
            losses = trainer.train_step(batch)
            trainer.step += 1
            if trainer.step % tcfg.get("log_every", 100) == 0:
                trainer.log(losses, t0)
            if trainer.step % tcfg.get("ckpt_every", 5000) == 0:
                trainer.save_checkpoint("latest")
        trainer.save_checkpoint("final")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
