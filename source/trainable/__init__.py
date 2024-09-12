from ..trainer import Trainable
from .trainable_mnist import TrainableMnist, TrainableMnistCfg

# This should be a union of all Trainable configuration types.
TrainableCfg = TrainableMnistCfg


def get_trainable_type(cfg: TrainableCfg) -> type[Trainable]:
    # To add a new Trainable:
    # 1. Add it to the dictionary below.
    # 2. Update DatasetCfg above.
    return {
        "mnist": TrainableMnist,
    }[cfg.name]
