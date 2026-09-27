import click
import mlflow
from .training.train import train
from .training.eval import evaluate
from .config import TrainConfig
import os


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
@click.option("--use-wandb", is_flag=True, help="Enable wandb (default: disable)")
def run_training(env_id: str, log_dir: str, use_wandb: bool):
    mlflow.set_tracking_uri("hoge")
    mlflow.set_experiment("mjc_rl_test_1")
    with mlflow.start_run() as _:
        train(env_id=env_id, log_dir_str=f"{log_dir}/{env_id}", config=TrainConfig())


@cli.command(name="eval")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
@click.option("--seed", default=0, help="seed")
def run_evaluation(env_id: str, log_dir: str, seed: int):
    evaluate(
        env_id=env_id,
        log_dir=f"{log_dir}/{env_id}",
        n_episodes=5,
        config=TrainConfig(),
        record_video=True,
    )


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_USERNAME"] = "hoge"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "hoge"
    cli()
