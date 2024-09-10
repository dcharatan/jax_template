from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import PRNGKeyArray
from optax import GradientTransformation
from grain.python import DataLoader

D = TypeVar("D")  # data type
C = TypeVar("C")  # config type


class Trainable(eqx.Module, Generic[C, D], ABC):
    cfg: C

    @abstractmethod
    def train_step(self, batch: D, rng: PRNGKeyArray) -> None:
        pass

    @abstractmethod
    def configure_optimizers(self) -> GradientTransformation:
        pass


@dataclass
class TrainerCfg:
    num_steps: int


def train(
    rng: PRNGKeyArray,
    trainable: Trainable[C, D],
    loader: DataLoader,
) -> None:
    a = 1
