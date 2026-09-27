from flax import nnx
import jax
import jax.numpy as jnp


class RunningStats(nnx.Module):
    """
    観測情報の正規化
    性能に与える影響が極めて大きいらしい
    訓練中にデータの平均・分散を記録し、その情報を使って推論時にデータを正規化することで学習を安定させる
    この実装は Welford's online algorithmっていうメモリ効率のいい手法
    """

    def __init__(self, in_features: int):
        """
        mean/m2/countは学習中に更新されるため、
        nnx.Variableで囲んでNNXの状態として管理する必要がある
        """
        self.in_features: int = in_features
        # 平均と差の累積
        self.mean = nnx.Variable(jnp.zeros(self.in_features, dtype=jnp.float32))
        # 分散計算用の二乗和(Welford's algirithm用)
        self.m2 = nnx.Variable(jnp.zeros(self.in_features, dtype=jnp.float32))
        # サンプル数
        self.count = nnx.Variable(0)

    @property
    def std(self) -> jax.Array:
        """
        標準偏差を返す
        まだデータがない時は1を返す
        """
        return jnp.where(
            jnp.bool(self.count),
            jnp.sqrt(self.m2 / self.count) + 1e-8,
            jnp.ones_like(self.mean.value),
        )

    def update(self, x: jax.Array):
        """
        バッチの平均・分散を計算し、既存の統計に合成
        """
        x = x.reshape(-1, self.in_features)
        batch_mean = jnp.mean(x, axis=0)
        batch_count = x.shape[0]
        batch_m2 = jnp.var(x, axis=0) * batch_count
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / new_count)
        new_m2 = self.m2 + batch_m2 + delta**2 * (self.count * batch_count) / new_count

        self.mean.value = new_mean
        self.m2.value = new_m2
        self.count.value = new_count
