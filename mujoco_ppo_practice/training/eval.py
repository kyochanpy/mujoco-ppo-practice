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
import numpy as np
import functools
from mujoco_playground._src.gait import draw_joystick_command
import imageio


def evaluate(
    env_id: str,
    n_episodes: int,
    log_dir: str,
    config: TrainConfig,
    record_video: bool = True,
):
    (env, env_cfg, obs_dim, _, action_dim, env_reset_fn, env_step_fn) = create_env(
        env_id
    )

    # Load the trained policy
    abstract_model = nnx.eval_shape(
        lambda: SquashedGauusianPolicy(
            obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0)
        )
    )
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir: Path = (Path(log_dir) / config.ckpt_dir).resolve()
    _graphdef, _abstract_state = nnx.split(abstract_model)
    _state = checkpointer.restore(ckpt_dir, _abstract_state)
    policy_nn = nnx.merge(_graphdef, _state)

    scores = []
    rng = jax.random.PRNGKey(config.seed)
    for n in range(n_episodes):
        trajectory, modify_scene_fns = [], []
        rng, key = jax.random.split(rng)
        state = env_reset_fn(key)
        for _ in range(env_cfg.episode_length):  # type: ignore
            rng, subkey = jax.random.split(rng)
            action, _, _ = policy_nn.sample_action(  # type: ignore
                obs=state.obs["state"].reshape(1, -1), key=subkey
            )
            state = env_step_fn(state, action[0])
            trajectory.append(state)
            if record_video:
                xyz = np.array(state.data.xpos[env._torso_body_id])  # type: ignore
                xyz += np.array([0, 0, 0.2])
                x_axis = state.data.xmat[env._torso_body_id, 0]  # type: ignore
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,  # 矢印の描画位置
                        theta=yaw,
                        scl=abs(state.info["command"][0]) / env_cfg.command_config.a[0],  # type: ignore
                    )
                )
            if state.done:
                break

        score: float = sum([s.reward for s in trajectory])
        scores.append(score)
        print(f"Episode {n + 1}, {len(trajectory)} steps, total reward: {score:.2f}")

        if record_video:
            print("Saving video...")
            frames: list[np.ndarray] = env.render(  # type: ignore
                trajectory,
                camera="track",
                height=240 * 3,
                width=380 * 3,
                modify_scene_fns=modify_scene_fns,
            )
            imageio.mimsave(  # type: ignore
                f"{log_dir}/eval_{n + 1}.mp4",
                frames,  # type: ignore
                fps=1 / env.dt,
            )

    return np.mean(scores)
