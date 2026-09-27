from typing import Mapping
from brax.envs.base import Wrapper
from jax._src.pjit import JitWrapped
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry, wrapper
import jax
import functools


def create_env(
    env_id: str, num_envs: int = 1
) -> tuple[MjxEnv | Wrapper, ConfigDict, int, int, int, JitWrapped, JitWrapped]:
    env_config = registry.get_default_config(env_id)
    env = registry.load(env_id, config=env_config)
    randomizer = registry.get_domain_randomizer(env_id)

    """
    obs_dim / priv_obs_dim / action_dim
    それぞれ観測や行動の次元数
    環境や制御対象によって変わるのでenvから読み込む
    """
    obs_size = env.observation_size
    if not isinstance(obs_size, Mapping):
        raise TypeError(f"Unsupported observation_size type: {type(obs_size)!r}")
    # obs_dim
    if isinstance(state := obs_size.get("state"), tuple):
        obs_dim = state[0]
    else:
        raise TypeError(f"Unsupported state type: {type(state)!r}")
    # priv_obs_dim
    if isinstance(privileged_state := obs_size.get("privileged_state"), tuple):
        priv_obs_dim = privileged_state[0]
    else:
        raise TypeError(f"Unsupported priv_obs_dim type: {type(privileged_state)!r}")
    # action_dim
    action_dim = env.action_size

    """
    並列で回す場合のセットアップ
    """
    if num_envs > 1:
        # jaxの乱数キーを環境分に分割
        # 各サブ環境が独立した乱数を使えるようにする
        keys = jax.random.split(jax.random.PRNGKey(42), num_envs)
        v_randomization_fn = functools.partial(randomizer, rng=keys)  # type: ignore
        env = wrapper.wrap_for_brax_training(
            env=env,
            episode_length=1000,
            action_repeat=1,
            randomization_fn=v_randomization_fn,
        )

    """
    reset/stepのjitコンパイル
    """
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    return env, env_config, obs_dim, priv_obs_dim, action_dim, reset_fn, step_fn
