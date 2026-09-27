from flax import nnx
from ..models.policy import SquashedGauusianPolicy
from ..models.value import ValueNN
import jax
import jax.numpy as jnp
from ..config import TrainConfig


@nnx.jit(static_argnames=("config",))
def train_step(
    batch_data: dict,
    policy_nn: SquashedGauusianPolicy,
    value_nn: ValueNN,
    policy_optimizer: nnx.Optimizer,
    value_optimizer: nnx.Optimizer,
    key: jax.random.PRNGKey,  # type: ignore
    config: TrainConfig,
):
    advantages = batch_data["advantages"]
    normalized_advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1e-8
    )

    def policy_loss_fn(policy_nn: SquashedGauusianPolicy) -> jax.Array:
        mu, std = policy_nn(batch_data["obs"])
        new_log_probs = policy_nn.log_prob(batch_data["raw_actions"], loc=mu, scale=std)

        ratio = jnp.exp(new_log_probs - batch_data["old_log_probs"])
        surr1 = ratio * normalized_advantages
        surr2 = (
            jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
            * normalized_advantages
        )
        policy_loss = -1 * jnp.minimum(surr1, surr2)

        entropy = policy_nn.entropy(mu, std, key)
        entropy_loss = -1 * config.entropy_coef * entropy

        loss = jnp.mean(policy_loss + entropy_loss)

        return loss

    def value_loss_fn(value_nn: ValueNN) -> jax.Array:
        values = value_nn(batch_data["obs_privileged"])
        value_loss = (values - batch_data["target_values"]) ** 2
        loss = jnp.mean(value_loss)
        return loss

    policy_loss, policy_grad = nnx.value_and_grad(policy_loss_fn)(policy_nn)
    policy_optimizer.update(policy_nn, policy_grad)

    value_loss, value_grad = nnx.value_and_grad(value_loss_fn)(value_nn)
    value_optimizer.update(value_nn, value_grad)

    return policy_loss, value_loss
