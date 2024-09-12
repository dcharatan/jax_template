from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from optax import GradientTransformation

from ..dataset.interface import Batch
from ..trainer import Trainable


class CNN(eqx.Module):
    """The model from the Equinox MNIST example:
    https://docs.kidger.site/equinox/examples/mnist/#the-model
    """

    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(3, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "3 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass(frozen=True)
class TrainableMnistCfg:
    name: Literal["mnist"]


class TrainableMnist(Trainable[Batch]):
    cfg: TrainableMnistCfg
    model: CNN

    def __init__(
        self,
        cfg: TrainableMnistCfg,
        rng: PRNGKeyArray,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = CNN(rng)

    def train_step(self, batch: Batch, rng: PRNGKeyArray) -> Float[Array, ""]:
        # In Equinox, layers/models are assumed to use scalar types, so vmap is
        # necessary here.
        pred_y = jax.vmap(self.model)(batch["rgb"])

        def cross_entropy(
            y: Int[Array, " batch"],
            pred_y: Float[Array, "batch 10"],
        ) -> Float[Array, ""]:
            # y are the true targets, and should be integers 0-9.
            # pred_y are the log-softmax'd predictions.
            pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
            return -jnp.mean(pred_y)

        return cross_entropy(batch["label"], pred_y)

    def configure_optimizer(self) -> GradientTransformation:
        return optax.adam(0.001)
