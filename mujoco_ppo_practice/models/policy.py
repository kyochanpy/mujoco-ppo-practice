from flax import nnx
import jax.numpy as jnp
import jax
from ..algo.running_stats import RunningStats


class SquashedGauusianPolicy(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.action_dim = action_dim

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

        self.mu = nnx.Linear(
            in_features=128,
            out_features=action_dim,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))

        self.running_stats = RunningStats(in_features=obs_dim)

    def __call__(self, obs, debug=False):
        x = (obs - self.running_stats.mean) / self.running_stats.std
        # siluは活性化関数
        x = nnx.silu(self.dense_1(x))
        x = nnx.silu(self.dense_2(x))
        x = nnx.silu(self.dense_3(x))
        mu = self.mu(x)
        std = (nnx.softplus(self.log_std.value) + 0.01) * jnp.ones_like(mu)
        return mu, std

    @nnx.jit
    def sample_action(
        self, obs, key: jax.random.PRNGKey  # type: ignore
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert obs.ndim == 2, "Input must be (batch_size, obs_dim)"
        mu, std = self(obs)
        raw_action = mu + std * jax.random.normal(key, shape=mu.shape)
        action = nnx.tanh(raw_action)
        log_prob = self.log_prob(raw_action, mu, std)
        return action, raw_action, log_prob

    def log_prob(self, raw_action, loc, scale):
        log_prob_normal = -0.5 * (
            jnp.square((raw_action - loc) / scale) + jnp.log(2 * jnp.pi * scale**2)
        ).sum(axis=-1, keepdims=True)

        # log(1 - tanh(x)^2) を数値的に安定した形で計算
        log_det_jacobian = 2.0 * (
            jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
        ).sum(axis=-1, keepdims=True)

        log_prob = log_prob_normal - log_det_jacobian
        return log_prob

    def entropy(self, loc, scale, key: jax.random.PRNGKey):  # type: ignore
        # tanh(Normal) のエントロピーは解析的に計算できないのでサンプリングして近似
        raw_action_sampled = loc + scale * jax.random.normal(key, shape=loc.shape)

        entropy_normal = 0.5 * (1 + jnp.log(2 * jnp.pi * scale**2)).sum(
            axis=-1, keepdims=True
        )
        # log(1 - tanh(x)^2) を数値的に安定した形で計算
        log_det_jacobian = 2.0 * (
            jnp.log(2.0)
            - raw_action_sampled
            - jax.nn.softplus(-2.0 * raw_action_sampled)
        ).sum(axis=-1, keepdims=True)
        entropy = entropy_normal + log_det_jacobian
        return entropy
