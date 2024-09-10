from dataclasses import dataclass
from typing import Literal

from jax import Array
from optax import GradientTransformation

from ..dataset.interface import Batch
from ..trainer import Trainable


@dataclass
class TrainableMnistCfg:
    name: Literal["mnist"]


class TrainableMnist(Trainable[TrainableMnistCfg, Batch]):
    def __init__(self, cfg: TrainableMnistCfg) -> None:
        super().__init__(cfg)

    def train_step(self, batch: Batch, rng: Array) -> None:
        raise NotImplementedError()

    def configure_optimizers(self) -> GradientTransformation:
        raise NotImplementedError()
