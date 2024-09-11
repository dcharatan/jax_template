from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from optax import GradientTransformation

from ..dataset.interface import Batch
from ..trainer import Trainable


@dataclass(frozen=True)
class TrainableMnistCfg:
    name: Literal["mnist"]


class TrainableMnist(Trainable[Batch]):
    cfg: TrainableMnistCfg
    dummy: Float[Array, "3 3"]

    def __init__(self, cfg: TrainableMnistCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dummy = jnp.ones((3, 3))

    def train_step(self, batch: Batch, rng: PRNGKeyArray) -> Float[Array, ""]:
        return (self.dummy**2).mean()

    def configure_optimizer(self) -> GradientTransformation:
        return optax.adam(0.001)
