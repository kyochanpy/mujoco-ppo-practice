import shutil
from pathlib import Path
from flax import nnx
from mujoco_playground import State
from tqdm import tqdm
from ..models.policy import SquashedGauusianPolicy
from ..models.value import ValueNN
import optax
from ..env import create_env
from ..config import TrainConfig
import jax
from typing import cast
import jax.numpy as jnp
from ..algo.advantage import compute_advantage_and_target
import mlflow
from ..algo.update import train_step
import orbax.checkpoint as ocp
from .eval import evaluate

def train(env_id: str, log_dir_str: str, config: TrainConfig) -> None:
    log_dir = Path(log_dir_str)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    (env, env_config, obs_dim, priv_obs_dim, action_dim, env_reset_fn, env_step_fn) = create_env(env_id, config.num_envs)

    policy_nn = SquashedGauusianPolicy(
        obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0)
    )
    policy_optimizer = nnx.Optimizer(
        policy_nn,
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=1e-4),
        ),
        wrt=nnx.Param
    )

    value_nn = ValueNN(obs_dim=priv_obs_dim, rngs=nnx.Rngs(0))
    value_optimizer = nnx.Optimizer(
        value_nn,
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=3e-4),
        ),
        wrt=nnx.Param
    )

    rng, *subkeys = jax.random.split(jax.random.PRNGKey(config.seed), config.num_envs + 1)
    # 初期状態を作る
    state = cast(State, env_reset_fn(jnp.array(subkeys)))
    trajectory: list[State] = [state]
    selected_actions: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    for i in tqdm(range(1, 1 + config.time_steps // config.num_envs)):
        rng, subkey = jax.random.split(rng)
        # action, raw_action, log_prob = policy_nn.sample_action(   # type: ignore
        #     obs=state.obs["state"], key=subkey
        # )
        action, raw_action, log_prob = cast(
            tuple[jax.Array, jax.Array, jax.Array],
            policy_nn.sample_action(state.obs["state"], key=subkey),  # type: ignore
        )
        selected_actions.append((action, raw_action, log_prob))

        state = cast(State, env_step_fn(state, action))
        trajectory.append(state)

        if i % config.unroll_length == 0:
            assert len(trajectory) == config.unroll_length + 1
            assert len(selected_actions) == config.unroll_length

            rewards = jnp.stack([s.reward for s in trajectory[1:]], axis=1)
            dones = jnp.stack([s.done for s in trajectory[1:]], axis=1)

            advantages, target_values = compute_advantage_and_target(
                value_nn,
                obs=jnp.stack([s.obs["privileged_state"] for s in trajectory], axis=1),
                rewards=rewards,
                dones=dones,
                config=config
            )

            B, T = config.num_envs, config.unroll_length
            batch_data = {
                "obs": jnp.stack(
                    [s.obs["state"] for s in trajectory[:-1]], axis=1
                ).reshape(B * T, -1),
                "obs_privileged": jnp.stack(
                    [s.obs["privileged_state"] for s in trajectory[:-1]], axis=1
                ).reshape(B * T, -1),
                "raw_actions": jnp.stack(
                    [a[1] for a in selected_actions], axis=1
                ).reshape(B * T, -1),
                "old_log_probs": jnp.stack(
                    [a[2] for a in selected_actions], axis=1
                ).reshape(B * T, 1),
                "advantages": advantages.reshape(B * T, 1),
                "target_values": target_values.reshape(B * T, 1),
            }

            for _ in range(config.num_update_per_batch):
                rng, subkey1, subkey2 = jax.random.split(rng, 3)
                indices = jax.random.permutation(subkey1, B * T)[:config.batch_size]
                _batch_data = jax.tree_util.tree_map(lambda x: x[indices], batch_data)

                ploss, vloss = train_step(
                    batch_data=_batch_data,
                    policy_nn=policy_nn,
                    value_nn=value_nn,
                    policy_optimizer=policy_optimizer, # type: ignore
                    value_optimizer=value_optimizer, # type: ignore
                    key=subkey2,
                    config=config
                )

            # Update running mean and variance of observations
            policy_nn.running_stats.update(batch_data["obs"])
            value_nn.running_stats.update(batch_data["obs_privileged"])

            mlflow.log_metrics(
                {
                    "ploss": ploss,  # type: ignore
                    "vloss": vloss,  # type: ignore
                    "reward": float(rewards.sum(axis=-1).mean()),
                },
                step=i * config.num_envs,
            )

            trajectory = trajectory[-1:]  # Keep the last state for the next rollout
            selected_actions = []  # Reset actions for the next rollout

        if i % 4000 == 0:
            # Save the model checkpoint
            checkpointer = ocp.StandardCheckpointer()
            ckpt_dir: Path = Path(log_dir / config.ckpt_dir).resolve()
            _, _state = nnx.split(policy_nn)
            checkpointer.save(ckpt_dir, _state, force=True)

            # Evaluate
            test_score = evaluate(
                env_id=env_id,
                n_episodes=5,
                log_dir=log_dir, # type: ignore
                config=config,
                record_video=True,
            )
            mlflow.log_metrics(
                {"episode_reward": float(test_score)},
                step=i * config.num_envs,
            )
