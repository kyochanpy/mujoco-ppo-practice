from flax import nnx
from ..config import TrainConfig
import jax
import jax.numpy as jnp


@nnx.jit(static_argnames=("config",))
def compute_advantage_and_target(value_nn, obs, rewards, dones, config: TrainConfig):
    """Computes advantage(GAE) and value targets."""
    B, T, _ = obs.shape
    values = value_nn(obs.reshape(B * T, -1)).reshape(B, T)
    values_t, values_t_plus_1 = values[:, :-1], values[:, 1:]
    deltas = rewards + config.discount * (1 - dones) * values_t_plus_1 - values_t

    def gae_scan_fn(advantage_plus_1, data_t):
        delta_t, done_t = data_t
        advantage_t = (
            delta_t
            + config.discount * config.gae_lambda * (1 - done_t) * advantage_plus_1
        )
        return advantage_t, advantage_t

    initial_carry = jnp.zeros(B)
    _, gae_T = jax.lax.scan(
        gae_scan_fn, initial_carry, (deltas.T, dones.T), reverse=True
    )
    gae = gae_T.T
    target_values = gae + values_t

    return gae, target_values
