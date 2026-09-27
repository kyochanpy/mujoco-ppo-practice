from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    time_steps: int = 200_000_000
    num_envs: int = 4096
    batch_size: int = 256
    unroll_length: int = 20
    num_update_per_batch: int = 320

    discount: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ckpt_dir: str = "checkpoints"
    seed: int = 0
