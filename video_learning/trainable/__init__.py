from ..trainer import Trainable
from .trainable_mnist import TrainableMnist, TrainableMnistCfg

TrainableCfg = TrainableMnistCfg


def get_trainable(cfg: TrainableCfg) -> Trainable:
    return {"mnist": TrainableMnist}[cfg.name](cfg)
