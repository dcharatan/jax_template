from dataclasses import dataclass

import jax
from grain.python import (
    BatchOperation,
    DataLoader,
    IndexSampler,
    PyGrainDatasetIterator,
    ShardOptions,
)

from .dataset_mnist import DatasetMnist, DatasetMnistCfg

DatasetCfg = DatasetMnistCfg


@dataclass
class DataLoaderCfg:
    per_device_batch_size: int
    worker_count: int
    worker_buffer_size: int
    num_epochs: int | None


def get_dataset_iterator(
    dataset_cfg: DatasetCfg,
    data_loader_cfg: DataLoaderCfg,
) -> PyGrainDatasetIterator:
    dataset = {"mnist": DatasetMnist}[dataset_cfg.name](dataset_cfg)
    data_source = dataset.get_data_source()

    # Combine the dataset's transformations with two batching operations: one for the
    # on-device batch dimension and one for the per-device batch dimension.
    operations = [
        *dataset.get_transformations(),
        BatchOperation(data_loader_cfg.per_device_batch_size, True),
        BatchOperation(jax.local_device_count(), True),
    ]

    loader = DataLoader(
        data_source=data_source,
        worker_count=data_loader_cfg.worker_count,
        worker_buffer_size=data_loader_cfg.worker_buffer_size,
        operations=operations,
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
    return iter(loader)
