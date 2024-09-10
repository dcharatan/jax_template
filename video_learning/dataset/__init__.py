from dataclasses import dataclass

import jax
from grain.python import DataLoader, IndexSampler, ShardOptions

from .dataset_mnist import DatasetMnist, DatasetMnistCfg

DatasetCfg = DatasetMnistCfg


@dataclass
class DataLoaderCfg:
    worker_count: int
    worker_buffer_size: int
    num_epochs: int | None


def get_data_loader(
    dataset_cfg: DatasetCfg,
    data_loader_cfg: DataLoaderCfg,
) -> DataLoader:
    dataset = {"mnist": DatasetMnist}[dataset_cfg.name](dataset_cfg)
    data_source = dataset.get_data_source()
    return DataLoader(
        data_source=data_source,
        worker_count=data_loader_cfg.worker_count,
        worker_buffer_size=data_loader_cfg.worker_buffer_size,
        operations=dataset.get_transformations(),
        sampler=IndexSampler(
            num_records=len(data_source),
            num_epochs=data_loader_cfg.num_epochs,
            shard_options=ShardOptions(
                shard_index=jax.process_index(),
                shard_count=jax.process_count(),
                drop_remainder=True,
            ),
            shuffle=False,
            seed=0,
        ),
    )
