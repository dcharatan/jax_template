from ..trainer import Trainable
from .trainable_mnist import TrainableMnist, TrainableMnistCfg

TrainableCfg = TrainableMnistCfg


def get_trainable_type(cfg: TrainableCfg) -> type[Trainable]:
    return {"mnist": TrainableMnist}[cfg.name]
