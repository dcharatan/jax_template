import sys
from dataclasses import dataclass

import hydra
from dacite import from_dict

from .dataset import DataLoaderCfg, DatasetCfg
from .trainable import TrainableCfg
from .trainer import TrainerCfg


@dataclass(frozen=True)
class RootCfg:
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    trainable: TrainableCfg
    trainer: TrainerCfg
    seed: int


def get_typed_config() -> RootCfg:
    # Read the configuration using Hydra.
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(config_name="main", overrides=sys.argv[1:])

    # Convert the configuration to a nested dataclass.
    return from_dict(RootCfg, cfg)
