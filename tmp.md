元コード → 新ファイルの移設マップ（そのまま移せる）
ppo_playground/env.py

create_env()

ppo_playground/models/running_stats.py

RunningStats

ppo_playground/models/policy.py

SquashedGaussianPolicy

ppo_playground/models/value.py

ValueNN

ppo_playground/algo/advantage.py

compute_advantage_and_target()

ppo_playground/algo/update.py

train_step()

ここは CLIP_EPS とか ENTROPY_COEF をどこに置くかで2択

(A) update.py 内に定数を残す（最小）

(B) ppo_playground/hparams.py を作って集約（少し綺麗）

ppo_playground/training/train.py

train()

ppo_playground/training/eval.py

evaluate()

ppo_playground/training/checkpoint.py

checkpoint保存・復元部分を関数化して移す

save_policy(policy_nn, ckpt_dir)

load_policy(obs_dim, action_dim, ckpt_dir) -> policy_nn

ppo_playground/training/logging.py（任意）

wandb.init, wandb.log, wandb.finish を薄く包む

ただし最初は train.py に直書きでもOK（分割しすぎない方が楽）

ppo_playground/cli.py

click の cli() と train/eval コマンド

中では training/train.py と training/eval.py を呼ぶだけ
