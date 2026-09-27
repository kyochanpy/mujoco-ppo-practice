from flax import nnx
from ..algo.running_stats import RunningStats


class ValueNN(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):
        self.dense_1 = nnx.Linear(
            in_features=obs_dim,
            out_features=512,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_2 = nnx.Linear(
            in_features=512,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_3 = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.out = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.running_stats = RunningStats(in_features=obs_dim)

    def __call__(self, obs):
        x = (obs - self.running_stats.mean) / self.running_stats.std
        x = nnx.silu(self.dense_1(x))
        x = nnx.silu(self.dense_2(x))
        x = nnx.silu(self.dense_3(x))
        out = self.out(x)
        return out
